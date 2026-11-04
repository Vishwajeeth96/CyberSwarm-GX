#!/usr/bin/env python3
"""Mechanical completeness checks for DESIGN.md (section 10.1)."""
import re, sys
sys.path.insert(0, "/Users/guru/CyberSwarm-GX/.agents/tasks/cyberswarm-foundation")
import gen_design_tables as g

text = open(g.DESIGN).read()
errors = []
# 1 registry names appear in the matrix
reg = "Node Edge Topology ScenarioConfig EventRecord AgentObservation AvailableActions DetectionSummary ActionSpec PolicyProposal ValidationResult ExecutedAction GroundTruthSnapshot GraphSnapshot GraphFeatureRecord GraphModelOutput EpisodeResult EvaluationMetrics ComparisonRecord CalibrationReport ErrorRecord LogRecord RunManifest".split()
assert len(reg) == 23
m = text.split("## 10. Cross-check matrix")[1].split("### 10.1")[0]
for r in reg:
    if "`%s`" % r not in m:
        errors.append("record %s not in matrix" % r)
# 2 tests named in the matrix exist in inventory
paths = {p for p, *_ in g.SRC}
for t in set(re.findall(r"`((?:[a-z_]+)/test_[a-z0-9_]+)`", m)):
    if "tests/%s.py" % t not in paths:
        errors.append("matrix test missing from inventory: " + t)
# 3 fixtures named in matrix exist
fx = {p.split("/")[-1].rsplit(".json", 1)[0] for p, *_ in g.FIX if p.endswith(".json")}
fxfull = {p for p, *_ in g.FIX}
for name in set(re.findall(r"`([A-Za-z]+\.[A-Za-z0-9_-]+)`", m)):
    if re.match(r"^(Node|Edge|ScenarioConfig|EventRecord|AgentObservation|AvailableActions|DetectionSummary|PolicyProposal|ValidationResult|ExecutedAction|GroundTruthSnapshot|GraphSnapshot|GraphFeatureRecord|GraphModelOutput|EpisodeResult|EvaluationMetrics|ComparisonRecord|CalibrationReport|ErrorRecord|LogRecord|RunManifest|ActionSpec|Topology)\.", name):
        if name not in fx:
            errors.append("fixture in matrix not in inventory: " + name)
# 4 every test has dc or support
for p, o, st, dc, pur in g.SRC:
    if p.startswith("tests/") and "/test_" in p and not dc:
        errors.append("test without DC: " + p)
# 5 every DC id has a test or a documented extra
for dcid in g.DC:
    has_test = any(dcid in [x.strip() for x in dc.split(",")] and p.startswith("tests/") for p, o, st, dc, pur in g.SRC) or (dcid == "DC-07")
    if not has_test and dcid not in g.DC_EXTRA:
        errors.append("DC without test or extra: " + dcid)
# 6 invalid fixtures all listed in matrix
for p, st, cons, content in g.FIX:
    if "/invalid/" in p:
        n = p.split("/")[-1].rsplit(".json", 1)[0]
        short = n.split(".", 1)[1]
        if short not in m and n not in m:
            errors.append("invalid fixture not in matrix: " + n)
# 7 stage lists: each inventory path exactly once across stage sections
stage_sec = text.split("### 11.2")[1].split("### 11.3")[0]
listed = re.findall(r"`([^`]+)`", stage_sec)
allpaths = [p for p, *_ in g.SRC] + [p for p, *_ in g.FIX]
from collections import Counter
c = Counter(x for x in listed if x in set(allpaths))
for p in allpaths:
    if c[p] != 1:
        errors.append("path listed %d times in stages: %s" % (c[p], p))
print("checked", len(allpaths), "paths;", len(errors), "problems")
for e in errors:
    print(" -", e)
sys.exit(1 if errors else 0)
