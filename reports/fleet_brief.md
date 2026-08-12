# Fleet Driving-Safety Brief

KMeans clustering on accelerometer and gyroscope window statistics produced two groups: a smaller, high-intensity cluster (Cluster 0, 51 windows) and a larger, low-intensity cluster (Cluster 1, 129 windows). Cluster 0 sits consistently above the dataset average on every motion-intensity feature, while Cluster 1 sits consistently below it, suggesting the algorithm has separated higher-motion driving windows from calmer ones. The separation is real but modest — an Adjusted Rand Index of 0.12 against withheld true labels means cluster assignment only weakly predicts the ground-truth driving-behavior class.

**Higher-risk cluster:** 0

## Evidence
- acc_mag_mean z-score = +1.168: average accelerometer magnitude is more than one standard deviation above the overall mean, indicating sustained higher force levels throughout these windows
- acc_mag_std z-score = +1.252: greater variability in accelerometer magnitude, reflecting more frequent or more pronounced changes in motion
- acc_mag_max z-score = +1.209: peak accelerometer magnitude values are substantially elevated, pointing to harder acceleration or braking events
- jerk_std z-score = +1.159: standard deviation of jerk is well above average, meaning the rate of change of acceleration is more erratic — a direct indicator of abrupt inputs
- jerk_rms z-score = +1.160: root-mean-square jerk is similarly elevated, confirming that high jerk is not just occasional but sustained across the window
- accX_std z-score = +0.993 and accX_max_abs z-score = +1.068: lateral accelerometer variability and peak lateral force are both roughly one standard deviation above average, consistent with sharper cornering or lane-change inputs
- accY_rate_below_neg3 z-score = +0.604: the fraction of samples where AccY drops below -3 m/s² is above average, indicating more frequent hard braking or deceleration events exceeding -3 m/s²
- accY_rate_above_pos3 z-score = +0.401: the fraction of samples where AccY exceeds +3 m/s² is also above average, consistent with more frequent hard acceleration events exceeding +3 m/s²
- gyroX_std z-score = +0.615, gyroY_std z-score = +0.571, gyroZ_std z-score = +0.680: all three gyroscope axes show above-average rotational variability, indicating more frequent or larger vehicle orientation changes across pitch, roll, and yaw

## Recommendation
Flag windows assigned to Cluster 0 for manual review in the telematics pipeline: because Cluster 0 is defined by elevated jerk (jerk_std z = +1.159, jerk_rms z = +1.160), peak lateral force (accX_max_abs z = +1.068), and elevated rates of AccY exceeding ±3 m/s², a threshold-based secondary screen on those three raw features within Cluster 0 windows can help prioritize which recorded sessions warrant driver coaching conversations, without waiting for fully labelled data.

## Caveats
- There are no vehicle or driver identifiers in this dataset, so cluster membership cannot be traced back to a specific car or individual driver, and no conclusions can be drawn about patterns across different vehicles or drivers.
- Each driving-behavior class (SLOW, NORMAL, AGGRESSIVE) was captured as a single continuous recorded session rather than a collection of independent trips, so these findings describe only the specific sessions that were recorded and should not be generalised to a broader driving population.
- Accelerometer and gyroscope signals alone cannot distinguish NORMAL from SLOW driving, because that difference is captured by vehicle speed — a quantity these inertial sensors do not measure — not by acceleration or rotation patterns.
