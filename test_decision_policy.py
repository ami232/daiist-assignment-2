"""Synthetic arithmetic fixtures only; these are not model performance results."""
import unittest
import numpy as np
from decision_policy import Costs,policy_metrics,choose_on_validation

class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.y=np.array([0,0,1,1]);self.scores=np.array([.1,.7,.4,.9])
        self.costs=Costs(false_positive=2,false_negative=20,review_per_case=1)
        self.inputs=dict(human_sensitivity=1,human_specificity=1,daily_volume=1000,
                         review_seconds=60,productive_seconds_per_person=6*3600)

    def test_prevalence_weighting_and_cost(self):
        r=policy_metrics(self.y,self.scores,.5,.5,self.costs,.1,**self.inputs)
        self.assertEqual(r['expected_confusion_per_1000'],{'TN':450,'FP':450,'FN':50,'TP':50})
        self.assertAlmostEqual(r['cost_eur_per_1000'],1900)
        self.assertEqual(r['review_share'],0)

    def test_review_band_cost_and_staffing(self):
        r=policy_metrics(self.y,self.scores,.3,.8,self.costs,.1,**self.inputs)
        self.assertEqual(r['expected_confusion_per_1000'],{'TN':900,'FP':0,'FN':0,'TP':100})
        self.assertAlmostEqual(r['cost_eur_per_1000'],500)
        self.assertAlmostEqual(r['review_cases_per_day'],500)
        self.assertAlmostEqual(r['people_required_fte'],500/360)

    def test_humans_are_not_assumed_perfect(self):
        inputs={**self.inputs,'human_sensitivity':.8,'human_specificity':.9}
        r=policy_metrics(self.y,self.scores,0,1.000001,self.costs,.1,**inputs)
        self.assertAlmostEqual(r['expected_confusion_per_1000']['FN'],20)
        self.assertAlmostEqual(r['expected_confusion_per_1000']['FP'],90)
        self.assertAlmostEqual(r['cost_eur_per_1000'],1580)

    def test_threshold_choice_respects_review_capacity(self):
        best,rows=choose_on_validation(self.y,self.scores,self.costs,.1,
             grid=[0,.3,.5,.8,1.000001],max_review_share=0,**self.inputs)
        self.assertEqual(best['review_share'],0)
        self.assertTrue(all(r['review_share']==0 for r in rows))
        self.assertEqual(best['cost_eur_per_1000'],min(r['cost_eur_per_1000'] for r in rows))

    def test_missing_class_is_rejected(self):
        with self.assertRaises(ValueError):
            policy_metrics([0,0],[.1,.2],.3,.8,self.costs,.1,**self.inputs)

    def test_missing_box_is_reviewed_even_with_a_low_damage_score(self):
        r=policy_metrics(self.y,self.scores,.5,.5,self.costs,.1,
                         force_review=[True,False,False,False],**self.inputs)
        self.assertAlmostEqual(r['review_share'],.45)
        self.assertAlmostEqual(r['expected_confusion_per_1000']['TN'],450)

    def test_promotion_requires_independent_evidence_and_approval(self):
        from feedback import promotion_gate
        champion=dict(evaluation_split='fresh_independent_audit',independent_audit_id='fixture-only',
            cost_eur_per_1000=100,damage_recall=.9,review_share=.1,capacity_share=.12,weights_sha256='old')
        candidate={**champion,'cost_eur_per_1000':90,'weights_sha256':'new'}
        self.assertFalse(promotion_gate(candidate,champion,'')['promote'])
        self.assertTrue(promotion_gate(candidate,champion,'test reviewer')['promote'])
        self.assertFalse(promotion_gate({**candidate,'damage_recall':.8},champion,'test reviewer')['promote'])
        with self.assertRaises(ValueError):
            promotion_gate({**candidate,'evaluation_split':'coursework_test'},champion,'test reviewer')

if __name__=='__main__':unittest.main()
