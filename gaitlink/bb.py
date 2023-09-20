class TurningDetection_NoCat:
    # ... (your existing class code)

    def post_process_turns(self):
        # Concatenate turns with less than 0.5 in between if they are facing in the same direction
        concatenated_turns = []
        if len(self.all_turns_) == 0:
            self.all_angles_, self.all_turns_, self.complete_turns_list_ = [], [], []
            return

        # Calculate endpoints of turns
        endpoints = np.sum(self.all_turns_, axis=1)
        # Extract starting points of turns
        startpoints = np.array(self.all_turns_)[:, 0]
        # Calculate distances between succeeding turns
        diffs = startpoints[1:] - endpoints[:-1]
        # Calculate indices of turns that might belong together
        concat_idx = np.where((diffs <= self.concat_length * self.fs))[0]

        ctr = 0
        for i, label in enumerate(self.all_turns_):
            # Turn has been processed already
            if i < ctr:
                continue
            while ctr in concat_idx:
                # Check if turns are facing in the same direction
                # Calculate integral values
                first = np.sum(self.data_[startpoints[ctr]: endpoints[ctr]])
                second = np.sum(self.data_[startpoints[ctr + 1]: endpoints[ctr + 1])
                # Check if they have the same sign
                if np.sign(first) == np.sign(second):
                    ctr += 1
                    continue
                break
            # Set new endpoint for elongated turn
            new_endpoint = endpoints[ctr]
            # Add new turn to list
            concatenated_turns.append([label[0], new_endpoint])
            ctr += 1

        # Exclude sections if the length is not suitable
        lengths = np.diff(concatenated_turns, axis=1) / self.fs
        suitable_index_list = [
            idx for idx, length in enumerate(lengths) if (self.min_length <= length <= self.max_length)
        ]
        self.all_turns_ = [concatenated_turns[t] for t in suitable_index_list]

        # Calculate turning angles and eliminate too small angles
        turning_angles = []
        for t in self.all_turns_:
            # Approximate integral by summing up the respective section
            integral = np.sum(self.gyr_z_lp[t[0]: t[1]] / self.fs)
            turning_angles.append(integral)
        self.all_angles_ = turning_angles

        # Store turns with turning angles within the range of angle threshold
        suitable_angles = np.where(
            (np.array(self.all_angles_) > self.angle_threshold[0])
            & (np.array(self.all_angles_) < self.angle_threshold[1])
        )[0]
        self.complete_turns_list_ = [self.all_turns_]
        self.suitable_angles_ = suitable_angles
