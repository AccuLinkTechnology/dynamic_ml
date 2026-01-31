//Experiment Overview
    - Laser stabilisation system.
    - A pole is controlled by a Raspberry Pi -> two motors, in pitch and twist (macro disturbance motors).
    - A compensatory "dynamic" stabilisation system: a Raspberry Pi controlling azimuth and elevation motors to counteract pole motion.
    - A laser mounted on the compensatory system pointing to a wall target.
    - A camera mounted on the pole pointing downward at a cross on the floor (not the laser).
    - A CNN running on a Jetson predicts required compensation deltas (Δaz, Δel) from camera images.
    - Jetson sends deltas via UDP to the Pi, which runs PID and moves compensatory motors.

//Target & Ground Truth
    - Target is to keep the laser stable on the wall despite pole motion.
    - Ground truth proxy: the cross on the floor as seen by the camera (camera does not observe the laser).
    - Labels: delta_azimuth and delta_elevation recorded during manual or scripted motor movements.
    - True zero reference was never globally defined; each sequence had its own implicit mechanical baseline.
    - Baseline mechanical offsets controlled between sequences; reset of pole using inclinometer on pole.
    - No IMU or inclinometer in the dynamic current system (camera-only perception).

//Dataset
    - Initially ~1400 images across 20 sequences. Some sequences smaller trhan others. (2, 3,4, 5 : 38-50 ish). 1 is 209. Rest vary between ~72 and ~100.
    - Each sequence has:
        - A reference frame (seqX_start 1.tga)
        - Multiple disturbed frames (seqX_N 1.tga)
        - CSV with delta_azimuth, delta_elevation.
    - Images are 320x180, normalized to [-1,1].
    - Current model uses DIFF input: (current_frame - reference_frame).
    - All four motors re-calibrated/zeroed between sequences.
    - Lighting conditions vary; cross position varies slightly; camera pose varies slightly between sequences.

//Known Faults (of previous 300/pic dataset).
    - Label bias: CNN outputs constant offsets (~ -1.6° az, -0.1° el) even when static.
    - CNN learned average mechanical misalignment rather than true disturbance correction.
    - Live system spirals because PID fights constant CNN bias.
    - Compensatory motors were commanded too frequently (misinterpreted 10Hz requirement).
    - No damping / rate limiting initially, causing runaway oscillation.
    - Camera reference frame drift (captured after system already misaligned).
    - Training data not balanced around zero-motion; too many biased “centered” samples.
    - CNN trained as absolute regressor instead of delta-from-reference control signal.

//Potential Fixes 
    - Pipeline-Specific
        -Altering pipeline to be more permissive: +-0.1 (due to the nature of a bullseye on a wall: anywhere in it is ok).
        -Not very good at picking up on crosses all the time? Maybe cross finding model.

    	- Altering Model
        - Re-center or re-label sequences: subtract per-sequence mean delta before training.
        - Train CNN to predict change-from-reference only (remove absolute motor offsets).
        - Use Huber loss for robustness to mislabeled samples.
     	- Maintain a library of multiple reference frames and average or dynamically select reference.
     	- Add diff amplification and gradient channels to improve sensitivity.
     	- Switch To Concat (if results point towards this).
    
    - Control-Specific    
    	- Live bias cancellation: average CNN output over 30s static period and subtract baseline bias.
    	- Reduce compensatory motor command rate (e.g., 2–5 Hz) with damping and slew limits.
       - Long-term: add IMU or encoder ground truth to decouple camera bias from mechanics. 

    
    - Collect balanced motion data around true zero (positive and negative deltas).
    - Add deadband to prevent micro-jitter corrections.
    - Implement manual and auto re-zero (capture new reference + bias reset hotkey).
