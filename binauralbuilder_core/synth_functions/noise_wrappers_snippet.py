
# ==========================================
# Synth Function Wrappers for Noise
# ==========================================

def noise_swept_notch(
    duration,
    sample_rate,
    lfo_freq=0.1,
    sweeps=None,
    notch_q=None,
    casc=None,
    start_lfo_phase_offset_deg=0.0,
    start_intra_phase_offset_deg=0.0,
    input_audio_path=None,
    noise_type="pink",
    lfo_waveform="sine",
    stereo_phase_invert=False,
    channels=2,
    static_notches=None,
    volume=1.0,
    **kwargs
):
    """
    Wrapper for _generate_swept_notch_arrays to be used as a synth function voice.
    Adapts the flat argument list to the specific arguments required by the internal generator.
    """
    # Ensure lists are properly formatted if passed as single values or None
    if sweeps is None: sweeps = [(1000.0, 10000.0)]
    if notch_q is None: notch_q = [25.0]
    if casc is None: casc = [10]
    
    # Generate the noise
    audio, _ = _generate_swept_notch_arrays(
        duration,
        sample_rate,
        lfo_freq,
        sweeps,
        notch_q,
        casc,
        start_lfo_phase_offset_deg,
        start_intra_phase_offset_deg,
        input_audio_path,
        noise_type,
        lfo_waveform,
        stereo_phase_invert,
        channels,
        static_notches
    )
    
    # Apply volume
    if volume != 1.0:
        audio *= volume
        
    return audio


def noise_swept_notch_transition(
    duration,
    sample_rate,
    start_lfo_freq=0.1,
    end_lfo_freq=0.1,
    start_sweeps=None,
    end_sweeps=None,
    start_q=None,
    end_q=None,
    start_casc=None,
    end_casc=None,
    start_lfo_phase_offset_deg=0.0,
    end_lfo_phase_offset_deg=0.0,
    start_intra_phase_offset_deg=0.0,
    end_intra_phase_offset_deg=0.0,
    input_audio_path=None,
    noise_type="pink",
    lfo_waveform="sine",
    initial_offset=0.0,
    transition_duration=None,
    transition_curve="linear",
    stereo_phase_invert=False,
    channels=2,
    static_notches=None,
    volume=1.0,
    **kwargs
):
    """
    Wrapper for _generate_swept_notch_arrays_transition to be used as a synth function voice.
    """
    # Ensure lists are properly formatted
    if start_sweeps is None: start_sweeps = [(1000.0, 10000.0)]
    if end_sweeps is None: end_sweeps = [(1000.0, 10000.0)]
    if start_q is None: start_q = [25.0]
    if end_q is None: end_q = [25.0]
    if start_casc is None: start_casc = [10]
    if end_casc is None: end_casc = [10]

    # Generate the noise
    audio, _ = _generate_swept_notch_arrays_transition(
        duration,
        sample_rate,
        start_lfo_freq,
        end_lfo_freq,
        start_sweeps,
        end_sweeps,
        start_q,
        end_q,
        start_casc,
        end_casc,
        start_lfo_phase_offset_deg,
        end_lfo_phase_offset_deg,
        start_intra_phase_offset_deg,
        end_intra_phase_offset_deg,
        input_audio_path,
        noise_type,
        lfo_waveform,
        initial_offset,
        transition_duration if transition_duration is not None else duration,
        transition_curve,
        stereo_phase_invert,
        channels,
        static_notches
    )

    # Apply volume
    if volume != 1.0:
        audio *= volume
        
    return audio
