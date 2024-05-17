from pykalman import KalmanFilter

def simple_kallman(lats, longs):
    """Applies simple Kallman filter on given measurements, returns tuple with ([lats], [longs])"""
    measurements = list(zip(lats, longs))

    initial_state_mean = [measurements[0][0], 0,
                        measurements[0][1], 0]
    transition_matrix = [[1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1]]
    observation_matrix = [[1, 0, 0, 0],
                        [0, 0, 1, 0]]

    kf = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean,
                    )
    kf = kf.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
    lats, _, longs, _ = list(zip(*smoothed_state_means))
    return lats, longs
