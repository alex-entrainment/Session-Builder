"""
PyInstaller hook for binauralbuilder_core package.

This hook ensures all synth_functions and utils submodules are included.
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all submodules
hiddenimports = collect_submodules('binauralbuilder_core')

# Ensure synth_functions are collected
hiddenimports += [
    'binauralbuilder_core.synth_functions.binaural_beat',
    'binauralbuilder_core.synth_functions.isochronic_tone',
    'binauralbuilder_core.synth_functions.qam_beat',
    'binauralbuilder_core.synth_functions.qam_beat_transition',
    'binauralbuilder_core.synth_functions.monaural_beat_stereo_amps',
    'binauralbuilder_core.synth_functions.dual_pulse_binaural',
    'binauralbuilder_core.synth_functions.spatial_ambi2d',
    'binauralbuilder_core.synth_functions.spatial_angle_modulation',
    'binauralbuilder_core.synth_functions.spatial_itd_stable',
    'binauralbuilder_core.synth_functions.stereo_am_independent',
    'binauralbuilder_core.synth_functions.wave_shape_stereo_am',
    'binauralbuilder_core.synth_functions.rhythmic_waveshaping',
    'binauralbuilder_core.synth_functions.fx_flanger',
    'binauralbuilder_core.synth_functions.noise_flanger',
    'binauralbuilder_core.synth_functions.subliminals',
    'binauralbuilder_core.synth_functions.common',
    'binauralbuilder_core.synth_functions.audio_engine',
    'binauralbuilder_core.synth_functions.sound_creator',
    'binauralbuilder_core.synth_functions.hybrid_qam_monaural_beat',
    'binauralbuilder_core.synth_functions.noise_wrappers_snippet',
]

# Collect data files if any
datas = collect_data_files('binauralbuilder_core')
