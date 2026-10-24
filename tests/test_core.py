import unittest, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core

class T(unittest.TestCase):
    def test_nfr01_all_urls_defanged(self):
        df = core.make(10); self.assertFalse(df.body_text.str.contains("https?://", regex=True).any())
    def test_fr03_holdout_campaigns_not_in_train(self):
        df = core.make(10); tr = df[~df.campaign_group.isin(core.HOLDOUT)]; self.assertFalse(set(tr.campaign_group) & core.HOLDOUT)
    def test_fr01_dedup_removes_copies(self):
        d, drop = core.dedup(core.make(60)); self.assertGreater(len(drop), 0)
    def test_fr04_abstain_band(self):
        self.assertEqual(core.verdict(.5, 0), "review"); self.assertEqual(core.verdict(.9, 0), "flag"); self.assertEqual(core.verdict(.1, 0), "clear")
    def test_at01_legit_reset_not_flagged_by_rules(self):
        self.assertLess(core.Rules().proba(["Your password was changed today. If this was you, no action is needed."])[0], .5)
if __name__ == "__main__": unittest.main(verbosity=2)
