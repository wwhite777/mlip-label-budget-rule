"""Exact synthetic paired-cluster AP calibration; no project data input."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from sklearn.metrics import average_precision_score
from confirm_runner.common import load_yaml, dump
from confirm_runner.statistics import add_maxima, read_families
from analysis.inference import p_up, p_down, holm

# Reproduced  fixture rates, with Monte Carlo tolerances; these are regression
# targets, not size-control thresholds. New fixtures are recorded separately below.
REGRESSION_RATES = {
    'tied_references': [.796,.588,.056,.021,.000,.000],
    'correlated_references': [.213,.105,.067,.028,.004,.000],
    'tied_contenders': [.077,.031,.076,.031,.033,.005],
    'regular_separated': [.062,.021,.061,.020,.043,.005],
    'single_boundary_null': [.069,.069],
    'noisy_contender_boundary': [.046,.046],
}


def regression_check(fixtures, outer):
    checks = []
    for fixture in fixtures:
        name = fixture['fixture']
        expected = REGRESSION_RATES.get(name)
        if expected is None:
            return False
        observed = list(fixture['rates'].values())
        tolerance = [4*np.sqrt(max(p*(1-p),1/outer)/outer)+1/outer for p in expected]
        checks.append(len(observed)==len(expected) and all(abs(x-p)<=t for x,p,t in zip(observed,expected,tolerance)))
    return bool(checks) and all(checks)


def boundary_fixtures(doc, outer, B, N=1000):
    """One F2 boundary null and two clearly rejected nulls, including a noisy arm."""
    results=[]; data={}
    patterns=((np.arange(4)[:,None]>>np.arange(2))&1)
    for name, q, seed in [('single_boundary_null', np.array([.04,0.]), 822210),
                          ('noisy_contender_boundary', np.array([.08,.04]), 822211)]:
        rng=np.random.default_rng(seed)
        probability=np.prod(np.where(patterns,q,1-q),axis=1)
        records=[]; counts=[]; witness=None; max_error=0.
        for iteration in range(outer):
            observed=rng.multinomial(N,probability)
            boot=rng.multinomial(N,observed/N,size=B)
            point=1-observed@patterns/(2*N); star=1-boot@patterns/(2*N)
            points={f'R0_{b}':point[0] for b in doc['backbones']['order']}
            stars={f'R0_{b}':star[:,0] for b in doc['backbones']['order']}
            for n in [300,1000,3000]:
                for b in doc['backbones']['order']:
                    for prefix in ['A5f_hist_gbm_','C1_ridge_']:
                        key=f'{prefix}{b}@{n}'
                        active=n==300 and prefix=='A5f_hist_gbm_' and b==doc['backbones']['order'][0]
                        points[key]=point[1] if active else .90
                        stars[key]=star[:,1] if active else np.full(B,.90)
                points[f'A4_hist_gbm_ensemble@{n}']=.90
                stars[f'A4_hist_gbm_ensemble@{n}']=np.full(B,.90)
            points['E0_mean']=.90;stars['E0_mean']=np.full(B,.90)
            rows=read_families(doc,add_maxima(doc,points,[300,1000,3000]),
                 add_maxima(doc,stars,[300,1000,3000]),[300,1000,3000],'P',synthetic=True)
            scalar=p_down(star[:,1]-star[:,0],doc['margin']['AP'])
            assert rows[0]['p_down']==scalar
            assert (rows[0]['p_down_holm']<=.05)==(scalar<=.05)  # Same false-decision events.
            assert all(r['reading']=='DOES_NOT_EXCEED' for r in rows[1:])
            records.append([scalar, rows[0]['p_down_holm']]);counts.append(observed)
            if iteration==0:
                # Validate the noisy contender shortcut with paired weighted AP.
                err=patterns[np.repeat(np.arange(4),observed)]; labels=np.tile([1,0],N)
                vrng=np.random.default_rng(seed+10000)
                for _ in range(10):
                    w=vrng.multinomial(N,np.ones(N)/N)
                    for j in range(2):
                        score=np.zeros(2*N);score[0::2]=np.where(err[:,j],-1.,1.)
                        actual=average_precision_score(labels,score,sample_weight=np.repeat(w,2))
                        max_error=max(max_error,abs(actual-(1-w@err[:,j]/(2*N))))
                witness=dict(production_F2=rows, contender_bootstrap_SD=float(np.std(star[:,1])),
                             weighted_AP_shortcut_max_error=max_error)
        records=np.asarray(records)
        rates=(records<=.05).mean(axis=0)
        results.append(dict(fixture=name,seed=seed,outer_datasets=outer,B=B,clusters=N,
            reference_error_probability=float(q[0]),contender_error_probability=float(q[1]),
            true_fixed_ORB_difference=.02,identical_contrasts=False,
            rates=dict(zip(['new_raw_down','new_Holm_down'],rates.tolist())),witness=witness))
        data[name+'_pvalues']=records;data[name+'_pattern_counts']=np.asarray(counts,dtype=np.int16)
        print(name,json.dumps(results[-1]['rates']),flush=True)
    return results,data


def calibrate(out, outer=1000, B=2000):
    started=time.monotonic();N=1000;m=.02
    doc=load_yaml()
    # Statistical calibration is independent of the external C05 eligibility gate.
    doc['execution']['C05_A5f_verdicts']={b:'PASS' for b in doc['backbones']['order']}
    patterns=((np.arange(32)[:,None]>>np.arange(5))&1)
    fixtures=[('tied_references',np.full(5,.04),0.,822204),
              ('correlated_references',np.full(5,.04),.75,822205),
              ('tied_contenders',np.array([.12,.10,.08,.06,.04]),0.,822206),
              ('regular_separated',np.array([.12,.10,.08,.06,.04]),0.,822207)]
    results=[];data={};max_error=0.
    for label,q,rho,seed in fixtures:
        rng=np.random.default_rng(seed)
        probability=np.prod(np.where(patterns,q,1-q),axis=1)
        if rho:
            common=np.zeros(32);common[0]=1-q[0];common[-1]=q[0]
            probability=(1-rho)*probability+rho*common
        records=[];counts=[];witness=None;original_consumed=False
        for iteration in range(outer):
            observed=rng.multinomial(N,probability)
            boot=rng.multinomial(N,observed/N,size=B)
            errors=observed@patterns;bstar=boot@patterns
            refs=1-errors/(2*N);ref_stars=1-bstar/(2*N)
            # The true best contender is perfect. In tied_contenders every contender
            # is perfect; in the regular case only one is, and the others are below it.
            old_down=(1+np.sum(bstar.min(axis=1)>=40))/(B+1)
            new_down=p_down(1-ref_stars[:,4],m)
            component_up=[p_up(1-ref_stars[:,j],m) for j in range(5)]
            iut=max(component_up)
            new_holm=holm([new_down]*3,['max@300','max@1000','max@3000'])
            up_holm=holm([iut]*6,[f'contrast{i}' for i in range(6)])
            records.append([old_down,min(1,3*old_down),new_down,new_holm[0],iut,up_holm[0]])
            counts.append(observed)
            # Match the check's RNG stream exactly, including its first-witness
            # weighted-AP validation draws (only on the tied-reference fixture).
            if label=='tied_references' and not original_consumed and 3*old_down<=.05:
                for _ in range(10):
                    rng.multinomial(N,np.ones(N)/N)
                original_consumed=True
            # Production path on every outer dataset, not only an isolated formula.
            points={f'R0_{b}':refs[j] for j,b in enumerate(doc['backbones']['order'])}
            stars={f'R0_{b}':ref_stars[:,j] for j,b in enumerate(doc['backbones']['order'])}
            for n in [300,1000,3000,10000]:
                for j,b in enumerate(doc['backbones']['order']):
                    value=1. if label=='tied_contenders' or j==0 else .90
                    for prefix in ['A5f_hist_gbm_','C1_ridge_']:
                        points[f'{prefix}{b}@{n}']=value;stars[f'{prefix}{b}@{n}']=np.full(B,value)
                points[f'A4_hist_gbm_ensemble@{n}']=.90;stars[f'A4_hist_gbm_ensemble@{n}']=np.full(B,.90)
            points['E0_mean']=.90;stars['E0_mean']=np.full(B,.90)
            rows=read_families(doc,add_maxima(doc,points,[300,1000,3000,10000]),
                add_maxima(doc,stars,[300,1000,3000,10000]),[300,1000,3000,10000],'P',synthetic=True)
            f2=[r for r in rows if r['family']=='F2']
            f1=next(r for r in rows if r['family']=='F1' and r['checkpoint']=='mace_medium')
            assert all(abs(r['p_down']-new_down)<1e-14 for r in f2)
            assert abs(f1['p_up']-iut)<1e-14
            assert all(r['reading']!='ADDS' for r in f2)
            if iteration==0:
                # Validate exact shortcut with actual synthetic weighted AP.
                pat=np.repeat(np.arange(32),observed);err=patterns[pat]
                labels=np.tile([1,0],N)
                verification_rng=np.random.default_rng(seed+10000)
                for _ in range(10):
                    w=verification_rng.multinomial(N,np.ones(N)/N)
                    for j in range(5):
                        score=np.zeros(2*N);score[0::2]=np.where(err[:,j],-1.,1.)
                        actual=average_precision_score(labels,score,sample_weight=np.repeat(w,2))
                        max_error=max(max_error,abs(actual-(1-w@err[:,j]/(2*N))))
                witness=dict(reference_error_counts=errors.tolist(),production_F2=f2,production_F1=f1)
        records=np.asarray(records)
        rates=(records<=.05).mean(axis=0)
        names=['old_raw_down','old_Holm_down','new_raw_down','new_Holm_down','F1_raw_IUT_up','F1_Holm_IUT_up']
        result=dict(fixture=label,seed=seed,outer_datasets=outer,B=B,clusters=N,
                    reference_error_probabilities=q.tolist(),reference_common_mixture=rho,
                    true_oracle_difference=.02,true_fixed_ORB_difference=.02,
                    contender_tie=label=='tied_contenders',null='down: d>=m; up: d<=m (boundary)',
                    rates=dict(zip(names,rates.tolist())),rejections=dict(zip(names,(records<=.05).sum(axis=0).tolist())),
                    identical_contrasts=True,witness=witness)
        results.append(result);data[label+'_pvalues']=records;data[label+'_pattern_counts']=np.asarray(counts,dtype=np.int16)
        print(label,json.dumps(result['rates']),flush=True)
    extra,extra_data=boundary_fixtures(doc,outer,B)
    results.extend(extra);data.update(extra_data)
    # Regular alternatives check direction/power without changing any calibration seed.
    power={}
    for direction,q in [('down',.0),('up',.08)]:
        probability=np.prod(np.where(patterns,q,1-q),axis=1)
        rng=np.random.default_rng(822208 if direction=='down' else 822209)
        hits=0
        for _ in range(outer):
            observed=rng.multinomial(N,probability);boot=rng.multinomial(N,observed/N,size=B)
            values=(boot@patterns)/(2*N)
            p=p_down(values[:,4],m) if direction=='down' else max(p_up(values[:,j],m) for j in range(5))
            hits+=p*(3 if direction=='down' else 6)<=.05
        power[direction]=hits/outer
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'calibration_data.npz',**data)
    report=dict(status='PASS' if regression_check(results,outer) else 'FAIL',synthetic_only=True,
                project_data_opened=False,outer_datasets_per_fixture=outer,B=B,alpha=.05,
                weighted_AP_shortcut_max_error=max_error,fixtures=results,regular_alternative_power=power,
                acceptance_scope='regression check: reproduced rates +/- four Monte Carlo SE plus one dataset; no size-control claim',
                regression_rates=REGRESSION_RATES,
                seconds=time.monotonic()-started,
                qualification='Monte Carlo validation of the paired empirical cluster bootstrap on specified synthetic fixtures; not a finite-sample proof or a forecast for CONFIRM. No refitting variance included.')
    dump(out/'calibration.json',report)
    lines=['# Synthetic calibration regression check (/ )','',
           f'Status: **{report["status"]}**. Each of six null fixtures uses {outer:,} independent datasets, {B:,} paired chemical-system bootstrap draws, 1,000 clusters and alpha = 0.05. No reduction of B. No project data or CONFIRM labels were opened.','',
           'Each cluster contains one positive and one negative. A reference misranks its positive with probability q; AP = 1 − q/2 exactly. A perfect contender has AP 1. At q_ORB = 0.04 the true oracle and fixed-ORB differences equal m = 0.02. The old rule subtracts the bootstrap reference maximum; the new down rule subtracts the fixed ORB reference. F1 uses max over the five component up p-values before six-contrast Holm.','',
           '| Fixture | Old raw down | Old Holm down | New raw down | New Holm down | F1 raw IUT up | F1 Holm IUT up |','|---|---:|---:|---:|---:|---:|---:|']
    names=['old_raw_down','old_Holm_down','new_raw_down','new_Holm_down','F1_raw_IUT_up','F1_Holm_IUT_up']
    for r in results:lines.append('| '+r['fixture']+' | '+' | '.join(f'{r["rates"][key]:.3f}' if key in r['rates'] else '—' for key in names)+' |')
    lines+=['', 'The first four family rates used identical contrasts at all three F2 budgets (and identical F1 IUT p-values). Their down rates therefore measure the same fixed-ORB statistic with Holm at its first step; reference correlation and contender ties do not vary that statistic. These rates are not evidence of general family-level error control. The gate is a regression check against the reproduced values +/- four Monte Carlo standard errors plus one dataset, not a requirement that family rates be <= 0.05. The tied-reference fixture preserves seed 822204 and its original RNG stream. Weighted-AP shortcut maximum discrepancy: '+str(max_error)+'.', '',
            'When the other contrasts are clearly rejected, the family error equals the scalar tail: approximately 0.056 at N = 1,000 and approximately 0.051 at N = 59,319 in the binary construction (exact binary-construction checks: 0.0563 and 0.0510). The new single-boundary-null fixture puts only max@300 on the boundary; max@1000 and max@3000 clearly reject. The noisy-contender boundary fixture bootstraps a contender with error probability 0.04 against a reference with error probability 0.08; both use the same paired cluster resamples. It also has only one boundary null. Neither fixture changes the decision algorithm.', '',
            'Correlated references use a 0.75 common-error mixture and otherwise independent errors. The tied-contender fixture ties all five A5f contenders at AP 1 with separated references. The regular case has a unique best reference and contender; remaining contenders have AP 0.90. All three F2 budgets share scores, as in the original counterexample. Production read_families is checked on every null dataset.', '',
            'Regular separated alternatives: down difference 0 and up difference 0.04; observed rejection fractions '+json.dumps(power)+'.', '',
            'These are fixture calibrations of approximate bootstrap tests, not a universal finite-sample guarantee. The IUT and fixed-reference sufficient-condition logic removes the invalid oracle maximum inference; validity of each scalar bootstrap component still relies on its usual regularity conditions. Empirical rates have Monte Carlo uncertainty (SE up to 0.0069 near alpha).', '',
            'The independent  unequal-cluster stress fixture yielded 0.370 family error (raw scalar also 0.370), with size-effective clusters far below the actual label-free CONFIRM size regime. Size metadata cannot establish outcome-dependent influence or calibration. accepts the quantified scalar approximation without a size correction. Oracle and contender-maximum percentile bands are descriptive, nominal percentile summaries; their coverage can be poor at ties and they must not be presented as calibrated 90 % confidence intervals.', '',
            'Data: calibration/calibration_data.npz stores all outer pattern counts and p-values (six for the original fixtures, two for the new boundary fixtures); calibration/calibration.json stores seeds, counts, first-dataset production witnesses, regression targets and timings. No bootstrap score matrices are retained.']
    (out.parent/'CALIBRATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--outer',type=int,default=1000);p.add_argument('--B',type=int,default=2000)
    a=p.parse_args();result=calibrate(a.out_root/'calibration',a.outer,a.B)
    raise SystemExit(result['status']!='PASS')
if __name__=='__main__':main()
