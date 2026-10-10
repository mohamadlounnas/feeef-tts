import os
import tempfile
import pytest
from src.inference.synthesize import DarijaTTSSynthesizer
from src.evaluation.evaluator import TTSEvaluator

def test_synthesizer():
    synth = DarijaTTSSynthesizer()
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_wav = os.path.join(tmp_dir, "test_out.wav")
        res = synth.synthesize(
            text="صحيت خويا، بارك الله فيك",
            output_path=out_wav,
            speaker="algiers_male_01",
            emotion="excited",
            speed=1.1
        )
        assert os.path.exists(out_wav)
        assert res["duration"] > 0
        assert res["emotion"] == "excited"

def test_evaluator():
    evaluator = TTSEvaluator()
    ref = "واش راك خويا"
    hyp = "واش راك خويا"
    cer = evaluator.calculate_cer(ref, hyp)
    assert cer == 0.0

    hyp_diff = "واش راك اختي"
    cer_diff = evaluator.calculate_cer(ref, hyp_diff)
    assert cer_diff > 0.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        synth = DarijaTTSSynthesizer()
        wav = os.path.join(tmp_dir, "eval.wav")
        synth.synthesize("صحيت خويا", wav)
        eval_res = evaluator.evaluate_synthesis("صحيت خويا", wav, "صحيت خويا")
        assert eval_res["cer"] == 0.0
        assert eval_res["snr_db"] > 10.0
