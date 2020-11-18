# code inspired from sktime ShapeletTransform 

from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

import heapq
import os
import time
import warnings
import numpy as np
from itertools import zip_longest
from operator import itemgetter
from sktime.utils.validation.series_as_features import check_X
from sktime.utils.data_container import detabularise, concat_nested_arrays
from sktime.transformers.series_as_features.base import BaseSeriesAsFeaturesTransformer
from sktime.transformers.series_as_features.shapelets import *

class OShapeletTransform(ShapeletTransform):
    def __init__(self,
                 min_shapelet_length=3,
                 max_shapelet_length=np.inf,
                 max_shapelets_to_store_per_class=200,
                 random_state=None,
                 verbose=0,
                 remove_self_similar=True,
                 nb_inst_per_class = 1,
                 predefined_ig_rejection_level = 0.05):
        super(OShapeletTransform, self).__init__(min_shapelet_length, 
                         max_shapelet_length,
                         max_shapelets_to_store_per_class,
                         random_state,
                         verbose,
                         remove_self_similar)
        self.nb_inst_per_class = nb_inst_per_class
        
        self.predefined_ig_rejection_level = predefined_ig_rejection_level
        
    def fit(self, X, y):
        """A method to fit the shapelet transform to a specified X and y
        Parameters
        ----------
        X: pandas DataFrame
            The training input samples.
        y: array-like or list
            The class values for X
        Returns
        -------
        self : FullShapeletTransform
            This estimator
        """
        X = check_X(X, enforce_univariate=True)

        if type(
                self) is ContractedShapeletTransform and \
                self.time_contract_in_mins <= 0:
            raise ValueError(
                "Error: time limit cannot be equal to or less than 0")

        X_lens = np.array([len(X.iloc[r, 0]) for r in range(len(
            X))])  # note, assumes all dimensions of a case are the same
        # length. A shapelet would not be well defined if indices do not match!
        X = np.array(
            [[X.iloc[r, c].values for c in range(len(X.columns))] for r in
             range(len(
                 X))])  # may need to pad with nans here for uneq length,
        # look at later

        num_ins = len(y)
        distinct_class_vals = \
            class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        candidates_evaluated = 0

        if type(self) is _RandomEnumerationShapeletTransform:
            num_series_to_visit = min(self.num_cases_to_sample, len(y))
        else:
            num_series_to_visit = num_ins

        shapelet_heaps_by_class = {i: ShapeletPQ() for i in
                                   distinct_class_vals}

        self.random_state = check_random_state(self.random_state)

        # Here we establish the order of cases to sample. We need to sample
        # x cases and y shapelets from each (where x = num_cases_to_sample
        # and y = num_shapelets_to_sample_per_case). We could simply sample
        # x cases without replacement and y shapelets from each case, but
        # the idea is that if we are using a time contract we may extract
        # all y shapelets from each x candidate and still have time remaining.
        # Therefore, if we get a list of the indices of the series and
        # shuffle them appropriately, we can go through the list again and
        # extract
        # another y shapelets from each series (if we have time).

        # We also want to ensure that we visit all classes so we will visit
        # in round-robin order. Therefore, the code below extracts the indices
        # of all series by class, shuffles the indices for each class
        # independently, and then combines them in alternating order. This
        # results in
        # a shuffled list of indices that are in alternating class order (
        # e.g. 1,2,3,1,2,3,1,2,3,1...)

        def _round_robin(*iterables):
            sentinel = object()
            return (a for x in zip_longest(*iterables, fillvalue=sentinel) for
                    a in x if a != sentinel)

        gen_ids_by_class = {i: self.random_state.permutation(np.where(y == i)[0])[:self.nb_inst_per_class] for i in
                             distinct_class_vals}
        case_ids_by_class = {i: np.where(y == i)[0] for i in distinct_class_vals}

        # if transform is random/contract then shuffle the data initially
        # when determining which cases to visit
        if type(self) is _RandomEnumerationShapeletTransform or type(
                self) is ContractedShapeletTransform:
            for i in range(len(distinct_class_vals)):
                self.random_state.shuffle(
                    case_ids_by_class[distinct_class_vals[i]])
                self.random_state.shuffle(
                    gen_ids_by_class[distinct_class_vals[i]])

        num_train_per_class = {i: len(case_ids_by_class[i]) for i in
                               case_ids_by_class}

        round_robin_case_order = _round_robin(
            *[list(v) for k, v in case_ids_by_class.items()])

        gen_round_robin_case_order = _round_robin(
            *[list(v) for k, v in gen_ids_by_class.items()])

        cases_to_visit = [(i, y[i]) for i in round_robin_case_order]

        gen_cases_to_visit = [(i, y[i]) for i in gen_round_robin_case_order]
        # this dictionary will be used to store all possible starting
        # positions and shapelet lengths for a give series length. This
        # is because we enumerate all possible candidates and sample without
        # replacement when assessing a series. If we have two series
        # of the same length then they will obviously have the same valid
        # shapelet starting positions and lengths (especially in standard
        # datasets where all series are equal length) so it makes sense to
        # store the possible candidates and reuse, rather than
        # recalculating each time

        # Initially the dictionary will be empty, and each time a new series
        # length is seen the dict will be updated. Next time that length
        # is used the dict will have an entry so can simply reuse
        possible_candidates_per_series_length = {}

        # a flag to indicate if extraction should stop (contract has ended)
        time_finished = False

        # max time calculating a shapelet
        # for timing the extraction when contracting
        start_time = time.time()

        def time_taken():
            return time.time() - start_time

        max_time_calc_shapelet = -1
        time_last_shapelet = time_taken()

        # for every series
        case_idx = 0
        while case_idx < len(gen_cases_to_visit):

            series_id = gen_cases_to_visit[case_idx][0]
            this_class_val = gen_cases_to_visit[case_idx][1]

            # minus 1 to remove this candidate from sums
            binary_ig_this_class_count = num_train_per_class[
                                             this_class_val] - 1
            binary_ig_other_class_count = (num_ins -
                                           binary_ig_this_class_count - 1)

            if self.verbose:
                if type(self) == _RandomEnumerationShapeletTransform:
                    print("visiting series: " + str(series_id) + " (#" + str(
                        case_idx + 1) + "/" + str(num_series_to_visit) + ")")
                else:
                    print("visiting series: " + str(series_id) + " (#" + str(
                        case_idx + 1) + ")")

            this_series_len = len(X[series_id][0])

            # The bound on possible shapelet lengths will differ
            # series-to-series if using unequal length data.
            # However, shapelets cannot be longer than the series, so set to
            # the minimum of the series length
            # and max shapelet length (which is inf by default)
            if self.max_shapelet_length == -1:
                this_shapelet_length_upper_bound = this_series_len
            else:
                this_shapelet_length_upper_bound = min(
                    this_series_len, self.max_shapelet_length)

            # all possible start and lengths for shapelets within this
            # series (calculates if series length is new, a simple look-up
            # if not)
            # enumerate all possible candidate starting positions and lengths.

            # First, try to reuse if they have been calculated for a series
            # of the same length before.
            candidate_starts_and_lens = \
                possible_candidates_per_series_length.get(
                    this_series_len)
            # else calculate them for this series length and store for
            # possible use again
            if candidate_starts_and_lens is None:
                candidate_starts_and_lens = [
                    [start, length] for start in
                    range(0, this_series_len - self.min_shapelet_length + 1)
                    for length in range(self.min_shapelet_length,
                                        this_shapelet_length_upper_bound + 1)
                    if start + length <= this_series_len]
                possible_candidates_per_series_length[
                    this_series_len] = candidate_starts_and_lens

            # default for full transform
            candidates_to_visit = candidate_starts_and_lens
            num_candidates_per_case = len(candidate_starts_and_lens)

            # limit search otherwise:
            if hasattr(self, "num_candidates_to_sample_per_case"):
                num_candidates_per_case = min(
                    self.num_candidates_to_sample_per_case,
                    num_candidates_per_case)
                cand_idx = list(self.random_state.choice(
                    list(range(0, len(candidate_starts_and_lens))),
                    num_candidates_per_case, replace=False))
                candidates_to_visit = [candidate_starts_and_lens[x] for x in
                                       cand_idx]

            for candidate_idx in range(num_candidates_per_case):

                # if shapelet heap for this class is not full yet, set entry
                # criteria to be the predetermined IG threshold
                ig_cutoff = self.predefined_ig_rejection_level
                # otherwise if we have max shapelets already, set the
                # threshold as the IG of the current 'worst' shapelet we have
                if shapelet_heaps_by_class[
                    this_class_val].get_size() >= \
                        self.max_shapelets_to_store_per_class:
                    ig_cutoff = max(
                        shapelet_heaps_by_class[this_class_val].peek()[0],
                        ig_cutoff)

                cand_start_pos = candidates_to_visit[candidate_idx][0]
                cand_len = candidates_to_visit[candidate_idx][1]

                candidate = ShapeletTransform.zscore(
                    X[series_id][:, cand_start_pos: cand_start_pos + cand_len])

                # now go through all other series and get a distance from
                # the candidate to each
                orderline = []

                # initialise here as copy, decrease the new val each time we
                # evaluate a comparison series
                num_visited_this_class = 0
                num_visited_other_class = 0

                candidate_rejected = False

                for comparison_series_idx in range(len(cases_to_visit)):
                    i = cases_to_visit[comparison_series_idx][0]

                    if y[i] != cases_to_visit[comparison_series_idx][1]:
                        raise ValueError("class match sanity test broken")

                    if i == series_id:
                        # don't evaluate candidate against own series
                        continue

                    if y[i] == this_class_val:
                        num_visited_this_class += 1
                        binary_class_identifier = 1  # positive for this class
                    else:
                        num_visited_other_class += 1
                        binary_class_identifier = -1  # negative for any
                        # other class

                    bsf_dist = np.inf

                    start_left = cand_start_pos
                    start_right = cand_start_pos + 1

                    if X_lens[i] == cand_len:
                        start_left = 0
                        start_right = 0

                    for num_cals in range(max(1, int(np.ceil((X_lens[
                                                                  i] -
                                                              cand_len) /
                                                             2)))):  # max
                        # used to force iteration where series len ==
                        # candidate len
                        if start_left < 0:
                            start_left = X_lens[i] - 1 - cand_len

                        comparison = ShapeletTransform.zscore(
                            X[i][:, start_left: start_left + cand_len])
                        dist_left = np.linalg.norm(candidate - comparison)
                        bsf_dist = min(dist_left * dist_left, bsf_dist)

                        # for odd lengths
                        if start_left == start_right:
                            continue

                        # right
                        if start_right == X_lens[i] - cand_len + 1:
                            start_right = 0
                        comparison = ShapeletTransform.zscore(
                            X[i][:, start_right: start_right + cand_len])
                        dist_right = np.linalg.norm(candidate - comparison)
                        bsf_dist = min(dist_right * dist_right, bsf_dist)

                        start_left -= 1
                        start_right += 1

                    orderline.append((bsf_dist, binary_class_identifier))
                    # sorting required after each add for early IG abandon.
                    # timsort should be efficient as array is almost in
                    # order - insertion-sort like behaviour in this case.
                    # Can't use heap as need to traverse in order multiple
                    # times, not just access root
                    orderline.sort()

                    if len(orderline) > 2:
                        ig_upper_bound = \
                            ShapeletTransform.calc_early_binary_ig(
                                orderline, num_visited_this_class,
                                num_visited_other_class,
                                binary_ig_this_class_count -
                                num_visited_this_class,
                                binary_ig_other_class_count -
                                num_visited_other_class)
                        # print("upper: "+str(ig_upper_bound))
                        if ig_upper_bound <= ig_cutoff:
                            candidate_rejected = True
                            break

                candidates_evaluated += 1
                if self.verbose > 3 and candidates_evaluated % 100 == 0:
                    print("candidates evaluated: " + str(candidates_evaluated))

                # only do if candidate was not rejected
                if candidate_rejected is False:
                    final_ig = ShapeletTransform.calc_binary_ig(
                        orderline,
                        binary_ig_this_class_count,
                        binary_ig_other_class_count)
                    accepted_candidate = Shapelet(series_id, cand_start_pos,
                                                  cand_len, final_ig,
                                                  candidate)

                    # add to min heap to store shapelets for this class
                    shapelet_heaps_by_class[this_class_val].push(
                        accepted_candidate)

                    # informal, but extra 10% allowance for self similar later
                    if shapelet_heaps_by_class[
                        this_class_val].get_size() > \
                            self.max_shapelets_to_store_per_class * 3:
                        shapelet_heaps_by_class[this_class_val].pop()

                # Takes into account the use of the MAX shapelet calculation
                # time to not exceed the time_limit (not exact, but likely a
                # good guess).
                if hasattr(self,
                           'time_contract_in_mins') and \
                        self.time_contract_in_mins \
                        > 0:
                    time_now = time_taken()
                    time_this_shapelet = (time_now - time_last_shapelet)
                    if time_this_shapelet > max_time_calc_shapelet:
                        max_time_calc_shapelet = time_this_shapelet
                    time_last_shapelet = time_now
                    if (
                            time_now + max_time_calc_shapelet) > \
                            self.time_contract_in_mins * 60:
                        if self.verbose > 0:
                            print(
                                "No more time available! It's been {0:02d}:{"
                                "1:02}".format(
                                    int(round(time_now / 60, 3)), int((round(
                                        time_now / 60, 3) - int(
                                        round(time_now / 60, 3))) * 60)))
                        time_finished = True
                        break
                    else:
                        if self.verbose > 0:
                            if candidate_rejected is False:
                                print(
                                    "Candidate finished. {0:02d}:{1:02} "
                                    "remaining".format(
                                        int(round(
                                            self.time_contract_in_mins -
                                            time_now / 60,
                                            3)),
                                        int((round(
                                            self.time_contract_in_mins -
                                            time_now / 60,
                                            3) - int(
                                            round((self.time_contract_in_mins
                                                   - time_now) / 60, 3))) *
                                            60)))
                            else:
                                print(
                                    "Candidate rejected. {0:02d}:{1:02} "
                                    "remaining".format(int(round(
                                        (self.time_contract_in_mins -
                                         time_now) / 60, 3)),
                                        int((round(
                                            (self.time_contract_in_mins -
                                             time_now) / 60,
                                            3) - int(
                                            round((self.time_contract_in_mins -
                                                   time_now) / 60, 3))) * 60)))

            # stopping condition: in case of iterative transform (i.e.
            # num_cases_to_sample have been visited)
            #                     in case of contracted transform (i.e. time
            #                     limit has been reached)
            case_idx += 1

            if case_idx >= num_series_to_visit:
                if hasattr(self,
                           'time_contract_in_mins') and time_finished is not \
                        True:
                    case_idx = 0
            elif case_idx >= num_series_to_visit or time_finished:
                if self.verbose > 0:
                    print("Stopping search")
                break

        # remove self similar here
        # for each class value
        #       get list of shapelets
        #       sort by quality
        #       remove self similar

        self.shapelets = []
        for class_val in distinct_class_vals:
            by_class_descending_ig = sorted(
                shapelet_heaps_by_class[class_val].get_array(),
                key=itemgetter(0), reverse=True)

            if self.remove_self_similar and len(by_class_descending_ig) > 0:
                by_class_descending_ig = \
                    ShapeletTransform.remove_self_similar_shapelets(
                        by_class_descending_ig)
            else:
                # need to extract shapelets from tuples
                by_class_descending_ig = [x[2] for x in by_class_descending_ig]

            # if we have more than max_shapelet_per_class, trim to that
            # amount here
            if len(by_class_descending_ig) > \
                    self.max_shapelets_to_store_per_class:
                max_n = self.max_shapelets_to_store_per_class
                by_class_descending_ig = by_class_descending_ig[:max_n]

            self.shapelets.extend(by_class_descending_ig)

        # final sort so that all shapelets from all classes are in
        # descending order of information gain
        self.shapelets.sort(key=lambda x: x.info_gain, reverse=True)
        self.is_fitted_ = True

        # warn the user if fit did not produce any valid shapelets
        if len(self.shapelets) == 0:
            warnings.warn(
                "No valid shapelets were extracted from this dataset and "
                "calling the transform method "
                "will raise an Exception. Please re-fit the transform with "
                "other data and/or "
                "parameter options.")

        self._is_fitted = True
        return self