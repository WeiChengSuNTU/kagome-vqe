r"""Every number quoted in the report or the deck, with its provenance.

:func:`collect` returns a registry ``key -> Number``. Writers turn it into
``report/tables/numbers.tex`` (one ``\V<key>`` macro per entry, e.g. ``\VExactE``) and
``slides/data/numbers.json`` (``{"ExactE": {"value": -18.0, "text": "−18.000", ...}}``).
Keys are letters only so they are valid TeX control-sequence names.

Two kinds of source exist. ``computed`` numbers are recomputed here from the package or read
from ``results/``. ``quoted`` numbers come from the repository's own prose (``docs/roadmap.md``)
because the code that produced them no longer exists (the pre-M0 seed bug); each names its
source and commit, and tests check they are not silently mixed with computed ones.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from docsgen import physics, runs
from docsgen.paths import REPORT_TABLES, ROOT, SLIDES_DATA
from kagome_vqe.results import load_run

__all__ = ["Number", "collect", "write_json", "write_tex"]

Kind = Literal["num", "int", "pct", "text"]
_KEY = re.compile(r"^[A-Za-z]+$")

METHODS = ("adam_forward", "adam_central", "adam_spsa", "adam_parameter_shift", "spsa")
METHOD_KEYS = {
    "adam_forward": "Fwd",
    "adam_central": "Cen",
    "adam_spsa": "Hyb",
    "adam_parameter_shift": "Ps",
    "spsa": "Spsa",
}
ANSATZ_KEYS = {"hva": "Hva", "rvb": "Rvb"}
STACKS = (
    "none",
    "readout",
    "symmetry",
    "readout+symmetry",
    "zne",
    "readout+zne",
    "readout+symmetry+zne",
    "twin",
    "readout+twin",
    "readout+symmetry+twin",
)
STACK_KEYS = {
    "none": "Raw",
    "readout": "Ro",
    "symmetry": "Sym",
    "readout+symmetry": "RoSym",
    "zne": "Zne",
    "readout+zne": "RoZne",
    "readout+symmetry+zne": "RoSymZne",
    "twin": "Twin",
    "readout+twin": "RoTwin",
    "readout+symmetry+twin": "RoSymTwin",
}
POINT_KEYS = {"m2-evaluation-0": "Zero", "m2-evaluation-0p3": "Pt"}
# M2 reruns with the corrected estimator (56aadd3), one error source at a time, and a fresh
# calibration and twin per repeat (2116da4)
SOURCE_KEYS = {
    "m2c-evaluation-0": "FullZero",
    "m2c-evaluation-0p3": "FullPt",
    "m2c_readout-evaluation-0p3": "RoPt",
    "m2c_gates-evaluation-0p3": "GatesPt",
}
EXACT_MODE_KEYS = {
    "raw": "Raw",
    "readout": "Ro",
    "symmetry": "Sym",
    "readout_symmetry": "RoSym",
    "correct_then_filter": "Ctf",
}
COMBO_KEYS = {
    "rvb_spsa": "RvbSpsa",
    "rvb_adam": "RvbAdam",
    "hva_spsa": "HvaSpsa",
    "hva_adam": "HvaAdam",
}


@dataclass(frozen=True)
class Number:
    """One quoted value.

    Attributes:
        value: The number (``nan`` for ``text`` kind).
        kind: ``num`` (fixed decimals), ``int``, ``pct`` (value is a fraction, shown in %), or
            ``text`` (a pre-formatted string in ``text``).
        decimals: Decimals shown for ``num`` and ``pct``.
        source: Where the value comes from (a results path, a function, or a document).
        quoted: True when copied from prose rather than recomputed.
        text: Display string for ``text`` kind.
    """

    value: float
    kind: Kind = "num"
    decimals: int = 3
    source: str = ""
    quoted: bool = False
    text: str = ""

    def plain(self) -> str:
        """Display string with a typographic minus (for HTML)."""
        if self.kind == "text":
            return self.text
        if self.kind == "int":
            s = f"{round(self.value)}"
        elif self.kind == "pct":
            s = f"{100 * self.value:.{self.decimals}f} %"
        else:
            s = f"{self.value:.{self.decimals}f}"
        if s.lstrip("-").strip("0. %") == "":
            s = s.lstrip("-")  # no negative zero
        return s.replace("-", "−")

    def math(self) -> str:
        """Display string for KaTeX in the deck (ASCII minus, escaped percent)."""
        if self.kind == "text":
            return f"\\text{{{self.text}}}"
        return self.plain().replace("\u2212", "-").replace(" %", "\\%")

    def tex(self) -> str:
        """Display string for LaTeX (math-mode minus, thin space before %)."""
        if self.kind == "text":
            return self.text
        if self.kind == "int":
            body = f"{round(self.value)}"
        elif self.kind == "pct":
            body = f"{100 * self.value:.{self.decimals}f}\\,\\%"
        else:
            body = f"{self.value:.{self.decimals}f}"
        if body.startswith("-") and body.lstrip("-").strip("0.\\,%") == "":
            body = body.lstrip("-")  # no negative zero
        return f"\\ensuremath{{{body}}}"


class Registry(dict[str, Number]):
    """``dict`` with a checked :meth:`add`."""

    def add(self, key: str, value: float, **kwargs: object) -> None:
        """Register ``key``; keys are unique and letters-only."""
        if not _KEY.match(key):
            msg = f"number key {key!r} must be letters only (it becomes a TeX macro)"
            raise ValueError(msg)
        if key in self:
            msg = f"duplicate number key {key!r}"
            raise KeyError(msg)
        self[key] = Number(float(value), **kwargs)  # type: ignore[arg-type]


def _per_iteration(run_path: str) -> tuple[float, int, int]:
    """(energies per iteration, maxiter, shots per energy) of one recorded run."""
    run = load_run(ROOT / run_path)
    maxiter = int(run.config.optimizer.maxiter)
    nfev = int(run.result["nfev"])
    shots = int(run.config.backend.shots)
    return (nfev - 1) / maxiter, maxiter, shots


def collect(*, recompute: bool = False) -> Registry:
    """Build the registry (reads ``results/``; simulator studies come from the cache)."""
    reg = Registry()
    _exact(reg)
    _noise(reg, recompute=recompute)
    _m0(reg)
    _m1(reg)
    _m2_fixed_point(reg)
    _m2_sources(reg)
    _m2_vqe(reg, "m2_vqe", "Mv")
    _m2_vqe(reg, "m2b_vqe", "Mw")
    _m2_exact(reg, recompute=recompute)
    _m2_seed_check(reg)
    return reg


def _exact(reg: Registry) -> None:
    f = physics.facts()
    src = "docsgen.physics.facts"
    reg.add("ExactE", f.spectrum[0], decimals=3, source=src)
    reg.add("ExactEone", f.spectrum[1], decimals=3, source=src)
    reg.add("ExactEtwo", f.spectrum[2], decimals=4, source=src)
    reg.add("ExactGap", f.spectrum[2] - f.spectrum[0], decimals=4, source=src)
    reg.add("NumSites", 12, kind="int", source=src)
    reg.add("NumBonds", f.num_bonds, kind="int", source=src)
    reg.add("NumTerms", f.num_terms, kind="int", source=src)
    reg.add("NumMatchings", len(f.matchings), kind="int", source=src)
    reg.add("SzSector", math.comb(12, 6), kind="int", source="binomial(12, 6)")
    reg.add("RingFracCycle", f.ring_fraction_cycle, kind="pct", decimals=0, source=src)
    reg.add("RingFracIdentity", f.ring_fraction_identity, kind="pct", decimals=0, source=src)
    reg.add("SingletIdentity", f.singlet_bonds_identity, kind="int", source=src)
    for key, value in f.e_zero.items():
        ansatz, order = key.split("_")
        reg.add(f"Ezero{ANSATZ_KEYS[ansatz]}{order.capitalize()}", value, decimals=1, source=src)
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        reg.add(f"Params{a}", f.num_parameters[ansatz], kind="int", source=src)
        reg.add(f"Cx{a}", f.cx[ansatz], kind="int", source=src)
        reg.add(f"Cx{a}Unshared", f.cx[f"{ansatz}_unshared"], kind="int", source=src)
        reg.add(f"Cx{a}NoLayout", f.cx[f"{ansatz}_nolayout"], kind="int", source=src)
        reg.add(f"Depth{a}", f.depth[ansatz], kind="int", source=src)
    reg.add(
        "HvaMinIdentity",
        physics.hva_minimum("identity"),
        decimals=2,
        source="docsgen.physics.hva_minimum",
    )
    reg.add("CoveringOverlap", f.covering_overlap, decimals=5, source=src)
    reg.add("MeasGroups", f.measurement_groups, kind="int", source=src)
    # parameter-shift cost: two energies per gate occurrence
    reg.add("PsEnergiesRvb", 2 * f.num_parameters["rvb"], kind="int", source="2 x 12 eSWAPs")
    reg.add(
        "PsEnergiesHva", 2 * 18 * f.num_parameters["hva"], kind="int", source="2 x 36 rotations"
    )


def _noise(reg: Registry, *, recompute: bool) -> None:
    nf = physics.noise_floor(recompute=recompute)
    src = "docsgen.physics.noise_floor (cache/noise_floor.json)"
    reg.add("NoiseShots", nf["shots"], kind="int", source=src)
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        e = nf["energies"]
        for label, key in (
            ("noise_free", "Ideal"),
            ("readout", "Readout"),
            ("gate", "Gate"),
            ("full", "Full"),
        ):
            reg.add(f"Floor{a}{key}", e[f"{ansatz}_{label}"], decimals=2, source=src)
        ideal = e[f"{ansatz}_noise_free"]
        r_ro = e[f"{ansatz}_readout"] / ideal
        r_g = e[f"{ansatz}_gate"] / ideal
        reg.add(f"Keep{a}Readout", r_ro, kind="pct", decimals=1, source=src)
        reg.add(f"Keep{a}Gate", r_g, kind="pct", decimals=1, source=src)
        reg.add(f"Keep{a}Product", ideal * r_ro * r_g, decimals=2, source=src)
    bc = physics.bound_compilation(recompute=recompute)
    src = "docsgen.physics.bound_compilation (cache/bound_compilation.json)"
    reg.add("BoundTheta", bc["theta"], decimals=1, source=src)
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        for label, word in (("parametric", "Param"), ("bound", "Bound")):
            row = bc["rows"][f"{ansatz}_{label}"]
            reg.add(f"{word}Cx{a}", row["cx"], kind="int", source=src)
            reg.add(f"{word}Depth{a}", row["depth"], kind="int", source=src)
            reg.add(f"{word}Keep{a}", row["retained"], kind="pct", decimals=1, source=src)
            reg.add(f"{word}Noisy{a}", row["noisy"], decimals=2, source=src)
        reg.add(f"BoundExact{a}", bc["rows"][f"{ansatz}_bound"]["exact"], decimals=2, source=src)
    sp = physics.shot_spread(recompute=recompute)
    src = "docsgen.physics.shot_spread (cache/shot_spread.json)"
    reg.add("SpreadRepeats", sp["repeats"], kind="int", source=src)
    for shots, word in (("1024", "Low"), ("4096", "High")):
        reg.add(f"Spread{word}Real", sp["real"][f"start_{shots}"], decimals=3, source=src)
        reg.add(f"Spread{word}Reported", sp["reported"][f"start_{shots}"], decimals=2, source=src)
        reg.add(f"Spread{word}Zero", sp["real"][f"zero_{shots}"], decimals=3, source=src)
    reg.add(
        "SpreadRatio",
        sp["reported"]["start_1024"] / sp["real"]["start_1024"],
        decimals=0,
        source=src,
    )


def _m0(reg: Registry) -> None:
    pre = runs.by_name("fake_guadalupe")
    post = runs.by_name("fake_guadalupe_m0")
    ideal = runs.by_name("statevector_m0")
    for combo, c in COMBO_KEYS.items():
        reg.add(
            f"PreFinal{c}",
            pre[combo]["energy"],
            decimals=2,
            source="results/fake_guadalupe-summary",
        )
        reg.add(
            f"PreBest{c}",
            pre[combo]["best_energy"],
            decimals=2,
            source="results/fake_guadalupe-summary",
        )
        reg.add(
            f"PostFinal{c}",
            post[combo]["final_energy"],
            decimals=2,
            source="results/fake_guadalupe_m0-summary",
        )
        reg.add(
            f"PostErr{c}",
            post[combo]["final_relative_error"],
            kind="pct",
            decimals=1,
            source="results/fake_guadalupe_m0-summary",
        )
        reg.add(
            f"PostBest{c}",
            post[combo]["best_energy"],
            decimals=2,
            source="results/fake_guadalupe_m0-summary",
        )
        reg.add(
            f"SvFinal{c}",
            ideal[combo]["final_energy"],
            decimals=4,
            source="results/statevector_m0-summary",
        )
        reg.add(
            f"SvErr{c}",
            ideal[combo]["final_relative_error"],
            kind="pct",
            decimals=3,
            source="results/statevector_m0-summary",
        )
    post_vals = [post[c]["final_energy"] for c in COMBO_KEYS]
    reg.add("PostFinalMin", min(post_vals), decimals=2, source="results/fake_guadalupe_m0-summary")
    reg.add("PostFinalMax", max(post_vals), decimals=2, source="results/fake_guadalupe_m0-summary")
    reg.add(
        "PostFinalShots",
        post["hva_spsa"]["final_shots"],
        kind="int",
        source="results/fake_guadalupe_m0-summary",
    )
    per_energy = [post[c]["wall_time_s"] / post[c]["nfev"] for c in COMBO_KEYS]
    reg.add(
        "SecPerEnergy",
        sum(per_energy) / len(per_energy),
        decimals=1,
        source="results/fake_guadalupe_m0-summary, wall time / evaluations",
    )
    # The pre-M0 seed-reuse bug: the code is gone, so these are quoted from the roadmap.
    src = "docs/roadmap.md section 1.3, measured on commit 81bee8b"
    reg.add("SeedBugSpread", 0.22, decimals=2, source=src, quoted=True)
    reg.add("SeedBugGrad", 2, kind="int", source=src, quoted=True)
    reg.add("SeedBugRows", 5, kind="int", source=src, quoted=True)
    # M0 spread check: measured with FakeGuadalupeV2 noise at the exact parameters (40 rows)
    src = "docs/roadmap.md M0 status, with device noise at the exact parameters"
    reg.add("RoadmapSpreadLow", 0.144, decimals=3, source=src, quoted=True)
    reg.add("RoadmapSpreadHigh", 0.060, decimals=3, source=src, quoted=True)


def _m1(reg: Registry) -> None:
    ideal = runs.aggregate("m1_ideal", "ideal_relative_error")
    noisy = runs.by_name("m1_noisy")
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        for method in METHODS:
            m = METHOD_KEYS[method]
            agg = ideal[f"{ansatz}_{method}"]
            src = "results/m1_ideal-summary (mean, sample std over seeds)"
            reg.add(f"MIdeal{a}{m}", agg.mean, kind="pct", decimals=3, source=src)
            reg.add(f"MIdeal{a}{m}Sd", agg.std, kind="pct", decimals=3, source=src)
            row = next(
                r for r in runs.summary("m1_ideal") if r["name"].endswith(f"{ansatz}_{method}-s1")
            )
            per_it, maxiter, shots = _per_iteration(row["path"])
            reg.add(f"MEnergies{a}{m}", per_it, kind="int", source=row["path"])
            if method == "adam_forward":
                reg.add(f"MIters{a}", maxiter, kind="int", source=row["path"])
            reg.add(f"MShots{a}{m}", shots, kind="int", source=row["path"])
            if ansatz == "rvb" and method in {
                k.split("-", 1)[-1].removeprefix("rvb_") for k in noisy
            }:
                nrow = noisy[f"rvb_{method}"]
                reg.add(
                    f"MNoisy{a}{m}",
                    nrow["ideal_relative_error"],
                    kind="pct",
                    decimals=3,
                    source="results/m1_noisy-summary",
                )
                reg.add(
                    f"MNoisyFinal{a}{m}",
                    nrow["final_energy"],
                    decimals=2,
                    source="results/m1_noisy-summary",
                )
    n_seeds = {agg.n for agg in ideal.values()}
    reg.add(
        "MSeeds",
        n_seeds.pop() if len(n_seeds) == 1 else float("nan"),
        kind="int",
        source="results/m1_ideal-summary",
    )
    first_noisy = next(iter(noisy.values()))
    _, noisy_iters, _ = _per_iteration(first_noisy["path"])
    reg.add("MNoisyIters", noisy_iters, kind="int", source=first_noisy["path"])
    # budget per iteration per basis, identical across methods by design
    reg.add(
        "MBudget",
        reg["MEnergiesHvaFwd"].value * reg["MShotsHvaFwd"].value,
        kind="int",
        source="energies per iteration x shots per energy",
    )
    # gradient noise per component in units of the per-energy shot noise sigma
    run = load_run(
        ROOT / next(r for r in runs.summary("m1_ideal") if "rvb_adam_forward" in r["name"])["path"]
    )
    eps = float(run.config.optimizer.fd_eps)
    reg.add("FdEps", eps, decimals=1, source="optimizer.fd_eps of the M1 runs")
    reg.add("SigmaFwd", math.sqrt(2) / eps, decimals=1, source="sqrt(2)/fd_eps")
    reg.add("SigmaCen", math.sqrt(2) / (2 * eps), decimals=1, source="sqrt(2)/(2 fd_eps)")
    reg.add("SigmaPsRvb", math.sqrt(2) / 2, decimals=2, source="sqrt(2)/2, one occurrence")
    reg.add("SigmaPsHva", math.sqrt(2 * 18) / 2, decimals=1, source="sqrt(36)/2, 18 occurrences")
    # the same at an equal shot budget per iteration: sigma = sigma_B * sqrt(m), m energies/iter
    for a in ("Hva", "Rvb"):
        n = reg[f"Params{a}"].value
        k_occ = 18 if a == "Hva" else 1
        m = {key: reg[f"MEnergies{a}{key}"].value for key in ("Fwd", "Cen", "Ps")}
        src = "per-component std x sqrt(energies per iteration), see report section M1"
        reg.add(f"BudgetFwd{a}", math.sqrt(2) / eps * math.sqrt(m["Fwd"]), decimals=1, source=src)
        reg.add(
            f"BudgetCen{a}", math.sqrt(2) / (2 * eps) * math.sqrt(m["Cen"]), decimals=1, source=src
        )
        reg.add(f"BudgetPs{a}", math.sqrt(k_occ / 2) * math.sqrt(m["Ps"]), decimals=1, source=src)
        if m["Fwd"] != n + 1 or m["Cen"] != 2 * n + 1:
            msg = f"unexpected evaluation counts for {a}: {m}"
            raise ValueError(msg)
    spsa_run = load_run(
        ROOT / next(r for r in runs.summary("m1_ideal") if "hva_spsa" in r["name"])["path"]
    )
    reg.add(
        "SpsaLr",
        float(spsa_run.config.optimizer.lr),
        decimals=2,
        source="optimizer.lr of the M1 SPSA runs",
    )
    reg.add(
        "SpsaCalibrated",
        0,
        kind="text",
        source="optimizer.calibrate of the M1 SPSA runs",
        text="calibrated" if spsa_run.config.optimizer.calibrate else "constant",
    )
    reg.add(
        "AdamLr", float(run.config.optimizer.lr), decimals=2, source="optimizer.lr of the M1 runs"
    )


def _stack_label(row: dict[str, object]) -> str:
    return (
        str(row["name"]).split("-", 1)[1].split("_", 1)[1].replace("_", "+").replace("raw", "none")
    )


def _stack_numbers(
    reg: Registry, name: str, prefix: Callable[[str], str]
) -> dict[str, dict[str, Any]]:
    """Mean, bias, spread and kept fraction of every stack in one fixed-point evaluation."""
    rows = runs.evaluation(name)
    src = f"results/{name}/evaluation.json"
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        for stack in STACKS:
            label = "raw" if stack == "none" else stack.replace("+", "_")
            row = rows[f"{ansatz}_{label}"]
            if _stack_label(row) != stack or row["mitigation"] != stack:
                msg = f"{src}: row {row['name']} is not stack {stack}"
                raise ValueError(msg)
            k = f"{prefix(a)}{STACK_KEYS[stack]}"
            reg.add(k, row["mean"], decimals=2, source=src)
            reg.add(f"{k}Bias", row["bias"], decimals=2, source=src)
            reg.add(
                f"{k}Rel",
                row["bias"] / abs(row["ideal_energy"]),
                kind="pct",
                decimals=2,
                source=src,
            )
            reg.add(f"{k}Spread", row["spread"], decimals=3, source=src)
            reg.add(
                f"{k}Below",
                (row["exact_energy"] - row["mean"]) / (row["spread"] / math.sqrt(row["repeats"])),
                decimals=1,
                source=f"{src}: (E0 - mean) / (spread / sqrt(repeats))",
            )
            reg.add(
                f"{k}Se",
                float(np.mean(row["reported_stds"])),
                decimals=3,
                source=f"{src}: mean reported standard error per energy",
            )
            reg.add(
                f"{k}SeRel",
                float(np.mean(row["reported_stds"])) / abs(row["ideal_energy"]),
                kind="pct",
                decimals=2,
                source=f"{src}: mean reported standard error / |noise-free energy|",
            )
            reg.add(
                f"{k}SemRel",
                row["spread"] / math.sqrt(row["repeats"]) / abs(row["ideal_energy"]),
                kind="pct",
                decimals=2,
                source=f"{src}: spread / sqrt(repeats) / |noise-free energy|",
            )
            if row.get("kept_fraction") is not None:
                reg.add(f"{k}Kept", row["kept_fraction"], kind="pct", decimals=0, source=src)
    return rows


def _m2_fixed_point(reg: Registry) -> None:
    """Each mitigation stack at fixed parameters (``kagome-vqe benchmark --point``)."""
    for name, p in POINT_KEYS.items():
        rows = _stack_numbers(reg, name, lambda a, p=p: f"Mt{a}{p}")
        for ansatz in ("hva", "rvb"):
            a = ANSATZ_KEYS[ansatz]
            src = f"results/{name}/evaluation.json"
            reg.add(f"MtIdeal{a}{p}", rows[f"{ansatz}_raw"]["ideal_energy"], decimals=2, source=src)
        reg.add(f"MtRepeats{p}", rows["hva_raw"]["repeats"], kind="int", source=src)
    rows = runs.evaluation("m2-evaluation-0")
    src = "results/m2-evaluation-0/evaluation.json"
    reg.add("MtShots", rows["hva_raw"]["shots_per_evaluation"], kind="int", source=src)
    reg.add("MtShotsZne", rows["hva_zne"]["shots_per_evaluation"], kind="int", source=src)
    for label, key in (
        ("readout", "Ro"),
        ("readout_zne", "RoZne"),
        ("twin", "Twin"),
        ("readout_twin", "RoTwin"),
    ):
        reg.add(f"MtSetup{key}", rows[f"hva_{label}"]["setup_shots"], kind="int", source=src)
    est = rows["hva_readout_symmetry"]["estimator"]
    e_mean = (sum(est["readout_e0"]) + sum(est["readout_e1"])) / (2 * len(est["readout_e0"]))
    reg.add("MtReadoutErr", e_mean, kind="pct", decimals=1, source=f"{src}: calibration")
    reg.add(
        "MtReadoutErrMax",
        max(est["readout_e0"] + est["readout_e1"]),
        kind="pct",
        decimals=1,
        source=f"{src}: calibration",
    )
    reg.add("MtKept", est["mean_kept_fraction"], kind="pct", decimals=0, source=src)
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        twin = rows[f"{ansatz}_twin"]["estimator"]
        reg.add(f"MtTwinIdeal{a}", twin["ideal_twin_energy"], decimals=2, source=src)
        reg.add(f"MtTwinNoisy{a}", twin["noisy_twin_energy"], decimals=2, source=src)
        reg.add(
            f"MtTwinFactor{a}",
            twin["ideal_twin_energy"] / twin["noisy_twin_energy"],
            decimals=3,
            source=src,
        )


def _m2_sources(reg: Registry) -> None:
    """Reruns with correct-then-filter, under full, readout-only and gate-only noise."""
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for name, sk in SOURCE_KEYS.items():
        rows[name] = _stack_numbers(reg, name, lambda a, sk=sk: f"Mb{a}{sk}")
        src = f"results/{name}/evaluation.json"
        for ansatz in ("hva", "rvb"):
            a = ANSATZ_KEYS[ansatz]
            reg.add(
                f"MbIdeal{a}{sk}",
                rows[name][f"{ansatz}_raw"]["ideal_energy"],
                decimals=2,
                source=src,
            )
            for stack, st in (("twin", "Raw"), ("readout_symmetry_twin", "Filt")):
                row = rows[name][f"{ansatz}_{stack}"]
                twin = row["estimator"]
                if not twin.get("fresh_setup_per_repeat"):
                    msg = f"{src}: {row['name']} reused one twin for every repeat"
                    raise ValueError(msg)
                draws = np.asarray(twin["noisy_twin_energy_per_setup"])
                reg.add(f"MbTwin{a}{sk}{st}", draws.mean(), decimals=3, source=f"{src}: mean twin")
                reg.add(
                    f"MbTwinSem{a}{sk}{st}",
                    draws.std(ddof=1) / math.sqrt(draws.size),
                    decimals=3,
                    source=f"{src}: standard error of the mean twin",
                )
                if st == "Filt":
                    # undo the rescaling repeat by repeat: the target energy each repeat measured
                    target = np.asarray(row["energies"]) * draws / twin["ideal_twin_energy"]
                    reg.add(f"MbTarget{a}{sk}", target.mean(), decimals=3, source=src)
                    reg.add(
                        f"MbTargetSem{a}{sk}",
                        target.std(ddof=1) / math.sqrt(target.size),
                        decimals=3,
                        source=f"{src}: energies x twin / ideal twin, per repeat",
                    )
            order = rows[name][f"{ansatz}_readout_symmetry"]["estimator"]["readout_symmetry_order"]
            if order != "correct_then_filter":
                msg = f"{src}: readout + symmetry ran as {order!r}, expected correct_then_filter"
                raise ValueError(msg)
    src = "results/m2c-*/evaluation.json"
    full = rows["m2c-evaluation-0"]
    reg.add("MbRepeats", full["hva_raw"]["repeats"], kind="int", source=src)
    reg.add("MbShots", full["hva_raw"]["shots_per_evaluation"], kind="int", source=src)
    reg.add("MbShotsZne", full["hva_zne"]["shots_per_evaluation"], kind="int", source=src)
    for label, key in (
        ("readout", "Ro"),
        ("readout_zne", "RoZne"),
        ("twin", "Twin"),
        ("readout_symmetry_twin", "RoSymTwin"),
    ):
        reg.add(f"MbSetup{key}", full[f"hva_{label}"]["setup_shots"], kind="int", source=src)
    reg.add(
        "MbTwinShots",
        full["hva_twin"]["estimator"]["twin_shots"],
        kind="int",
        source=f"{src}: shots per basis of the twin measurement",
    )
    chosen = [
        abs(reg[f"Mb{a}{sk}RoSymTwinRel"].value)
        for a in ANSATZ_KEYS.values()
        for sk in SOURCE_KEYS.values()
    ]
    reg.add("MbMaxRoSymTwin", max(chosen), kind="pct", decimals=2, source=f"{src}: max |rel. bias|")
    gated = [
        reg[f"Mb{a}{sk}RoSymTwinRel"].value
        for a in ANSATZ_KEYS.values()
        for sk in SOURCE_KEYS.values()
        if sk != "RoPt"
    ]
    note = f"{src}: readout + symmetry + twin, settings with gate noise"
    reg.add("MbRoSymTwinMin", min(gated), kind="pct", decimals=1, source=f"{note} (min)")
    reg.add("MbRoSymTwinMax", max(gated), kind="pct", decimals=1, source=f"{note} (max)")
    _m2_first_reruns(reg)
    _m2_zne_ranges(reg)
    for stack, st in (("readout_symmetry", "RoSym"), ("symmetry", "Sym")):
        kept = [
            rows[name][f"{ansatz}_{stack}"]["kept_fraction"]
            for name in ("m2c-evaluation-0", "m2c-evaluation-0p3", "m2c_gates-evaluation-0p3")
            for ansatz in ("hva", "rvb")
        ]
        reg.add(f"MbKeptMin{st}", min(kept), kind="pct", decimals=0, source=src)
        reg.add(f"MbKeptMax{st}", max(kept), kind="pct", decimals=0, source=src)
    ratios = [
        reg[f"Mb{a}{sk}RoSymSe"].value / reg[f"Mb{a}{sk}RoSymSpread"].value
        for a in ANSATZ_KEYS.values()
        for sk in SOURCE_KEYS.values()
    ]
    reg.add("MbSeRatioMin", min(ratios), decimals=1, source=f"{src}: bootstrap SE / spread")
    reg.add("MbSeRatioMax", max(ratios), decimals=1, source=f"{src}: bootstrap SE / spread")
    r = full["hva_raw"]["repeats"]
    reg.add(
        "MbSpreadRelErr",
        1 / math.sqrt(2 * (r - 1)),
        kind="pct",
        decimals=0,
        source=f"1 / sqrt(2 (R - 1)), relative error of a standard deviation from R = {r}",
    )


def _m2_first_reruns(reg: Registry) -> None:
    """The first reruns (m2b) shared one calibration and one twin across the repeats."""
    old = runs.evaluation("m2b-evaluation-0")
    osrc = "results/m2b-evaluation-0/evaluation.json"
    first = [
        abs(r["bias"]) / abs(r["ideal_energy"])
        for n in ("m2b-evaluation-0", "m2b-evaluation-0p3", "m2b_readout-evaluation-0p3",
                  "m2b_gates-evaluation-0p3")
        for k, r in runs.evaluation(n).items()
        if k.endswith("readout_symmetry_twin")
    ]  # fmt: skip
    reg.add("MoMaxRoSymTwin", max(first), kind="pct", decimals=2, source="results/m2b-*")
    r = old["hva_readout_symmetry_twin"]
    reg.add("MoHvaFullZeroRoSymTwinSpread", r["spread"], decimals=3, source=osrc)
    reg.add("MoHvaFullZeroRoSymTwinSe", float(np.mean(r["reported_stds"])), decimals=3, source=osrc)


def _m2_zne_ranges(reg: Registry) -> None:
    """Range of the ZNE stacks' bias over the reruns with gate noise."""
    src = "results/m2c-*/evaluation.json"
    for st in ("RoZne", "RoSymZne"):
        vals = [
            reg[f"Mb{a}{sk}{st}Rel"].value
            for a in ANSATZ_KEYS.values()
            for sk in SOURCE_KEYS.values()
            if sk != "RoPt"  # no gate noise there, so ZNE has nothing to extrapolate
        ]
        note = f"{src}: rel. bias over the settings with gate noise"
        reg.add(f"Mb{st}Min", min(vals), kind="pct", decimals=1, source=f"{note} (min)")
        reg.add(f"Mb{st}Max", max(vals), kind="pct", decimals=1, source=f"{note} (max)")


def _m2_vqe(reg: Registry, name: str, prefix: str) -> None:
    """Full optimisations with a mitigation stack (``Mv``: first study, ``Mw``: revised)."""
    vqe = runs.by_name(name)
    src = f"results/{name}-summary"
    for combo, key in (("hva_spsa", "HvaSpsa"), ("rvb_adam_parameter_shift", "RvbPs")):
        row = vqe[combo]
        reg.add(f"{prefix}Final{key}", row["final_energy"], decimals=3, source=src)
        reg.add(f"{prefix}Std{key}", row["final_std"], decimals=3, source=src)
        reg.add(
            f"{prefix}Err{key}", row["final_relative_error"], kind="pct", decimals=2, source=src
        )
        reg.add(f"{prefix}Ideal{key}", row["ideal_energy"], decimals=3, source=src)
        reg.add(
            f"{prefix}IdealErr{key}",
            row["ideal_relative_error"],
            kind="pct",
            decimals=2,
            source=src,
        )
        reg.add(f"{prefix}Shots{key}", row["total_shots"], kind="int", source=src)
        reg.add(f"{prefix}Nfev{key}", row["nfev"], kind="int", source=src)
        reg.add(
            f"{prefix}Below{key}",
            (row["exact_energy"] - row["final_energy"]) / row["final_std"],
            decimals=1,
            source=f"{src}: (E0 - final) / final_std",
        )
        run = load_run(ROOT / row["path"])
        reg.add(
            f"{prefix}Iters{key}", int(run.config.optimizer.maxiter), kind="int", source=row["path"]
        )
        reg.add(
            f"{prefix}ShotsPer{key}", int(run.config.backend.shots), kind="int", source=row["path"]
        )


def _m2_exact(reg: Registry, *, recompute: bool) -> None:
    """Infinite-shot bias of each counts-level estimator (``physics.mitigation_exact``)."""
    ex = physics.mitigation_exact(recompute=recompute)
    src = "docsgen.physics.mitigation_exact (cache/mitigation_exact.json)"
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        for point, p in ((0.0, "Zero"), (0.3, "Pt"), ("twin", "Twin")):
            exact = ex["exact"][f"{ansatz}_{point}"]
            if point == "twin":
                reg.add(f"MxIdeal{a}Twin", exact, decimals=2, source=src)
            for noise, n in (("full", "Full"), ("readout_only", "Ro"), ("gates_only", "Gt")):
                key = f"{ansatz}_{point}_{noise}"
                for mode, m in EXACT_MODE_KEYS.items():
                    e = ex["energies"][key][mode]
                    reg.add(f"Mx{a}{p}{n}{m}", e, decimals=2, source=src)
                    reg.add(
                        f"Mx{a}{p}{n}{m}Rel",
                        (e - exact) / abs(exact),
                        kind="pct",
                        decimals=1,
                        source=src,
                    )
                reg.add(f"Mx{a}{p}{n}Kept", ex["kept"][key], kind="pct", decimals=0, source=src)
                for mode, m in (("raw", "Raw"), ("correct_then_filter", "Ctf")):
                    reg.add(
                        f"MxDamp{a}{p}{n}{m}",
                        ex["energies"][key][mode] / exact,
                        decimals=3,
                        source=f"{src}: noisy / noise-free energy",
                    )
        # twin rescaling in the infinite-shot limit: what the chosen stack converges to
        twin = ex["exact"][f"{ansatz}_twin"]
        for point, p in ((0.0, "Zero"), (0.3, "Pt")):
            exact = ex["exact"][f"{ansatz}_{point}"]
            for noise, n in (("full", "Full"), ("readout_only", "Ro"), ("gates_only", "Gt")):
                for mode, m in (("raw", "Raw"), ("correct_then_filter", "Ctf")):
                    target = ex["energies"][f"{ansatz}_{point}_{noise}"][mode]
                    noisy_twin = ex["energies"][f"{ansatz}_twin_{noise}"][mode]
                    e = target * twin / noisy_twin
                    reg.add(
                        f"MxTw{a}{p}{n}{m}Rel",
                        (e - exact) / abs(exact),
                        kind="pct",
                        decimals=2,
                        source=f"{src}: E(target) x E_ideal(twin) / E(twin)",
                    )
    for a in ANSATZ_KEYS.values():
        for m in ("Raw", "Ctf"):
            d = [reg[f"MxDamp{a}{p}Gt{m}"].value for p in ("Zero", "Pt", "Twin")]
            reg.add(
                f"MxDampSpread{a}{m}",
                max(d) / min(d) - 1,
                kind="pct",
                decimals=1,
                source=f"{src}: max/min - 1 of the damping over target and twin, gate noise",
            )
    _m2_limits(reg, ex)
    tw = [
        abs(reg[f"MxTw{a}{p}{n}CtfRel"].value)
        for a in ANSATZ_KEYS.values()
        for p in ("Zero", "Pt")
        for n in ("Full", "Gt")
    ]
    reg.add("MxTwMin", min(tw), kind="pct", decimals=1, source=f"{src}: min |twin-rescaled bias|")
    reg.add("MxTwMax", max(tw), kind="pct", decimals=1, source=f"{src}: max |twin-rescaled bias|")
    # leading-order over-correction factor: filtered value times prod 1/(1 - e0 - e1) per pair
    k = ex["energies"]["hva_0.0_readout_only"]
    reg.add(
        "MxOverPredicted",
        1 / (1 - 2 * reg["MtReadoutErr"].value) ** 2,
        decimals=3,
        source="1 / (1 - 2 e)^2 with the mean calibrated flip probability e",
    )
    reg.add("MxOverFactor", k["readout_symmetry"] / k["symmetry"], decimals=3, source=src)
    reg.add(
        "MxOverPct",
        reg["MxOverFactor"].value - 1,
        kind="pct",
        decimals=1,
        source=f"{src}: over-correction factor - 1",
    )
    reg.add(
        "MxPredictFull",
        reg["MxOverFactor"].value * ex["energies"]["hva_0.0_full"]["symmetry"],
        decimals=2,
        source=f"{src}: over-correction factor x symmetry-only energy, full noise",
    )


def _m2_limits(reg: Registry, ex: dict[str, Any]) -> None:
    """Sampled revised stack against its infinite-shot limit, split into target and twin."""
    src = "docsgen.physics.mitigation_exact"
    limits = {"FullZero": ("Zero", "Full"), "FullPt": ("Pt", "Full"),
              "RoPt": ("Pt", "Ro"), "GatesPt": ("Pt", "Gt")}  # fmt: skip
    noise_name = {"Full": "full", "Ro": "readout_only", "Gt": "gates_only"}
    zs = {}
    for a, ansatz in ((ANSATZ_KEYS[x], x) for x in ANSATZ_KEYS):
        for sk, (p, n) in limits.items():
            k = f"Mb{a}{sk}RoSymTwin"
            z = (reg[f"{k}Rel"].value - reg[f"MxTw{a}{p}{n}CtfRel"].value) / reg[f"{k}SemRel"].value
            zs[(a, sk)] = z
            reg.add(
                f"MbZ{a}{sk}",
                z,
                decimals=1,
                source=f"{src} and results/m2c-*: (sampled - limit) / SEM",
            )
            point = 0.0 if p == "Zero" else 0.3
            tgt = ex["energies"][f"{ansatz}_{point}_{noise_name[n]}"]["correct_then_filter"]
            twn = ex["energies"][f"{ansatz}_twin_{noise_name[n]}"]["correct_then_filter"]
            reg.add(f"MbExactTarget{a}{sk}", tgt, decimals=3, source=src)
            reg.add(f"MbExactTwin{a}{sk}", twn, decimals=3, source=src)
            reg.add(
                f"MbTwinZ{a}{sk}",
                (reg[f"MbTwin{a}{sk}Filt"].value - twn) / reg[f"MbTwinSem{a}{sk}Filt"].value,
                decimals=1,
                source=f"{src}: (mean twin - exact twin) / SEM",
            )
            reg.add(
                f"MbTargetZ{a}{sk}",
                (reg[f"MbTarget{a}{sk}"].value - tgt) / reg[f"MbTargetSem{a}{sk}"].value,
                decimals=1,
                source=f"{src}: (mean target - exact target) / SEM",
            )
            # contributions to the relative bias of the rescaled energy (first order)
            reg.add(
                f"MbTwinShift{a}{sk}",
                1 - twn / reg[f"MbTwin{a}{sk}Filt"].value,
                kind="pct",
                decimals=1,
                source=f"{src}: 1 - exact twin / mean twin",
            )
            reg.add(
                f"MbTargetShift{a}{sk}",
                1 - reg[f"MbTarget{a}{sk}"].value / tgt,
                kind="pct",
                decimals=1,
                source=f"{src}: 1 - mean target / exact target",
            )
    reg.add("MbZMax", max(abs(z) for z in zs.values()), decimals=1, source=f"{src}: max |z|")
    # chance of such deviations if every setting were at its limit (Student t, R - 1 dof)
    from scipy import stats

    dof = reg["MbRepeats"].value - 1
    p_row = 2 * stats.t.sf(reg["MbZMax"].value, dof)
    reg.add(
        "MbPRow",
        1 - (1 - p_row) ** len(zs),
        kind="pct",
        decimals=0,
        source=f"{src}: P(some |z| >= max |z| among {len(zs)} settings), Student t",
    )
    # the two full-noise runs recorded the same twin draws; count that data set once
    for a in ANSATZ_KEYS.values():
        if reg[f"MbTwin{a}FullPtFilt"].value != reg[f"MbTwin{a}FullZeroFilt"].value:
            msg = f"{a}: full-noise twin draws differ between the two points"
            raise ValueError(msg)
    comps = [
        reg[f"Mb{part}Z{a}{sk}"].value
        for part in ("Target", "Twin")
        for a in ANSATZ_KEYS.values()
        for sk in limits
        if not (part == "Twin" and sk == "FullPt")
    ]
    cut = 2.5
    over = sum(abs(z) > cut for z in comps)
    reg.add("MbCompCut", cut, decimals=1, source="threshold for component z-scores")
    reg.add("MbCompN", len(comps), kind="int", source=f"{src}: target and twin z-scores")
    reg.add("MbCompOver", over, kind="int", source=f"{src}: components with |z| > {cut}")
    reg.add(
        "MbPComp",
        stats.binom.sf(over - 1, len(comps), 2 * stats.t.sf(cut, dof)),
        kind="pct",
        decimals=0,
        source=f"{src}: P(at least that many of {len(comps)} components with |z| > {cut})",
    )
    reg.add(
        "MbZOver",
        sum(abs(z) > 2 for z in zs.values()),
        kind="int",
        source=f"{src}: settings with |z| > 2",
    )


# HVA, readout + symmetry + twin, theta = 0, 16 fresh-set-up repeats per seed. Run by the
# development session through evaluate_fixed and not saved under results/, so quoted:
# (noise, seed) -> (row rel. bias, SEM, twin, SEM, target, SEM)
SEED_CHECK = {
    ("Full", "A"): (0.0132, 0.0022, -10.832, 0.019, -17.102, 0.024),
    ("Full", "B"): (0.0114, 0.0014, -10.806, 0.018, -17.091, 0.018),
    ("Gt", "A"): (0.0134, 0.0011, -10.834, 0.013, -17.102, 0.007),
    ("Gt", "B"): (0.0121, 0.0016, -10.823, 0.015, -17.105, 0.012),
}


def _m2_seed_check(reg: Registry) -> None:
    """Rerun of the one 2.7-sigma setting with new seeds (quoted; see ``SEED_CHECK``)."""
    src = "docs/roadmap.md at 1824789 and the dev session's report (seeds 1001, 2002)"
    reg.add("MqSeedA", 1001, kind="int", source=src, quoted=True)
    reg.add("MqSeedB", 2002, kind="int", source=src, quoted=True)
    reg.add("MqRepeats", 16, kind="int", source=src, quoted=True)
    for (n, seed), (row, row_sem, twin, twin_sem, target, target_sem) in SEED_CHECK.items():
        k = f"{n}{seed}"
        exact_twin = reg[f"MbExactTwinHva{'FullZero' if n == 'Full' else 'GatesPt'}"].value
        exact_target = reg[f"MxHvaZero{n}Ctf"].value
        reg.add(f"MqRow{k}", row, kind="pct", decimals=2, source=src, quoted=True)
        reg.add(f"MqRowSem{k}", row_sem, kind="pct", decimals=2, source=src, quoted=True)
        reg.add(f"MqTwin{k}", twin, decimals=3, source=src, quoted=True)
        reg.add(f"MqTwinSem{k}", twin_sem, decimals=3, source=src, quoted=True)
        reg.add(f"MqTarget{k}", target, decimals=3, source=src, quoted=True)
        reg.add(f"MqTargetSem{k}", target_sem, decimals=3, source=src, quoted=True)
        reg.add(f"MqTwinZ{k}", (twin - exact_twin) / twin_sem, decimals=1, source=src, quoted=True)
        reg.add(
            f"MqTargetZ{k}",
            (target - exact_target) / target_sem,
            decimals=1,
            source=src,
            quoted=True,
        )
        reg.add(
            f"MqRowZ{k}",
            (row - reg[f"MxTwHvaZero{n}CtfRel"].value) / row_sem,
            decimals=1,
            source=src,
            quoted=True,
        )


def write_tex(reg: Registry, path: Path = REPORT_TABLES / "numbers.tex") -> Path:
    r"""One ``\providecommand{\V<key>}{...}`` per number, with its source as a comment."""
    lines = ["% Generated by `uv run python -m docsgen`; do not edit.", ""]
    for key, num in sorted(reg.items()):
        flag = " [quoted]" if num.quoted else ""
        lines.append(f"\\providecommand{{\\V{key}}}{{{num.tex()}}}  % {num.source}{flag}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_json(reg: Registry, path: Path = SLIDES_DATA / "numbers.json") -> Path:
    """``{key: {value, text, source, quoted}}`` for the deck."""
    data = {
        key: {
            "value": None if math.isnan(n.value) else n.value,
            "text": n.plain(),
            "math": n.math(),
        }
        | {k: v for k, v in asdict(n).items() if k in ("source", "quoted")}
        for key, n in sorted(reg.items())
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    return path
