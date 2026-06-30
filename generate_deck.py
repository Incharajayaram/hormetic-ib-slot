"""Generate Lossfunk submission deck as a 16:9 landscape slide PDF."""
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus.flowables import Flowable
import reportlab.lib.colors as colors

# 16:9 slide dimensions
SW = 33.87 * cm
SH = 19.05 * cm
MARGIN_H = 1.6 * cm
MARGIN_V = 1.2 * cm

# Color palette — no red, varied
NAVY    = HexColor("#1e293b")   # primary dark
BLUE    = HexColor("#2563eb")   # accent / headings
TEAL    = HexColor("#0d9488")   # secondary accent
AMBER   = HexColor("#d97706")   # warnings / priorities
GREEN   = HexColor("#059669")   # positive / correct
SLATE   = HexColor("#475569")   # body text
LIGHT   = HexColor("#f1f5f9")   # row backgrounds
RULE    = HexColor("#cbd5e1")   # dividers
MID     = HexColor("#94a3b8")   # captions / secondary


def build_styles():
    s = getSampleStyleSheet()

    def add(name, **kw):
        if name not in s:
            s.add(ParagraphStyle(name=name, **kw))
        return s[name]

    add("CoverTitle",
        fontName="Helvetica-Bold", fontSize=32, textColor=NAVY,
        alignment=TA_CENTER, leading=40, spaceAfter=8)

    add("CoverSub",
        fontName="Helvetica-Bold", fontSize=20, textColor=BLUE,
        alignment=TA_CENTER, leading=26, spaceAfter=6)

    add("CoverMeta",
        fontName="Helvetica", fontSize=12, textColor=SLATE,
        alignment=TA_CENTER, leading=18, spaceAfter=4)

    add("SlideTitle",
        fontName="Helvetica-Bold", fontSize=24, textColor=NAVY,
        leading=30, spaceAfter=4)

    add("SlideSection",
        fontName="Helvetica-Bold", fontSize=13, textColor=TEAL,
        leading=17, spaceAfter=3, spaceBefore=8)

    add("Body",
        fontName="Helvetica", fontSize=11, textColor=SLATE,
        leading=15, spaceAfter=4)

    add("Bullet",
        fontName="Helvetica", fontSize=11, textColor=SLATE,
        leading=15, spaceAfter=3, leftIndent=14, bulletText="•",
        bulletIndent=0)

    add("BulletBold",
        fontName="Helvetica-Bold", fontSize=11, textColor=NAVY,
        leading=15, spaceAfter=1, leftIndent=14, bulletText="•",
        bulletIndent=0)

    add("SubBullet",
        fontName="Helvetica", fontSize=10, textColor=SLATE,
        leading=14, spaceAfter=2, leftIndent=26, bulletText="-",
        bulletIndent=14)

    add("Warn",
        fontName="Helvetica-Bold", fontSize=11, textColor=AMBER,
        leading=15, spaceAfter=2)

    add("Good",
        fontName="Helvetica-Bold", fontSize=11, textColor=GREEN,
        leading=15, spaceAfter=2)

    add("Caption",
        fontName="Helvetica-Oblique", fontSize=8, textColor=MID,
        alignment=TA_CENTER, leading=11, spaceAfter=2)

    add("ClosingBig",
        fontName="Helvetica-Bold", fontSize=13, textColor=NAVY,
        alignment=TA_CENTER, leading=19, spaceAfter=4)

    add("ClosingBody",
        fontName="Helvetica", fontSize=12, textColor=SLATE,
        alignment=TA_CENTER, leading=17, spaceAfter=0)

    return s


S = build_styles()


def sp(h=8):
    return Spacer(1, h)


def rule(color=RULE, thick=0.5):
    return HRFlowable(width="100%", thickness=thick, color=color,
                      spaceAfter=6, spaceBefore=2)


def accent_rule():
    return HRFlowable(width="100%", thickness=2, color=BLUE,
                      spaceAfter=8, spaceBefore=0)


def slide_header(title, subtitle=None):
    elems = [Paragraph(title, S["SlideTitle"]), accent_rule()]
    if subtitle:
        elems.append(Paragraph(subtitle, S["SlideSection"]))
    return elems


def b(text):
    return Paragraph(text, S["Bullet"])


def bb(text):
    return Paragraph(text, S["BulletBold"])


def sb(text):
    return Paragraph(text, S["SubBullet"])


def body(text):
    return Paragraph(text, S["Body"])


def sec(text):
    return Paragraph(text, S["SlideSection"])


def warn(text):
    return Paragraph(text, S["Warn"])


def good(text):
    return Paragraph(text, S["Good"])


def build_deck(out_path):
    doc = SimpleDocTemplate(
        out_path,
        pagesize=(SW, SH),
        leftMargin=MARGIN_H, rightMargin=MARGIN_H,
        topMargin=MARGIN_V, bottomMargin=MARGIN_V,
        title="Lossfunk Autoresearch — Hormetic IB Slot",
        author="Inchara J",
    )

    story = []
    CW = SW - 2 * MARGIN_H  # usable content width

    # ── COVER ────────────────────────────────────────────────────────────────
    story += [
        sp(0.8 * cm),
        HRFlowable(width="100%", thickness=4, color=BLUE, spaceAfter=16),
        Paragraph("Lossfunk Autoresearch Submission", S["CoverTitle"]),
        sp(6),
        Paragraph("Hormetic IB Scheduling for Persistent Object Identity",
                  S["CoverSub"]),
        sp(8),
        HRFlowable(width="100%", thickness=4, color=TEAL, spaceAfter=18),
        sp(6),
        Paragraph("Inchara J   |   incharajayaram2020@gmail.com", S["CoverMeta"]),
        Paragraph("github.com/Incharajayaram/hormetic-ib-slot", S["CoverMeta"]),
        Paragraph("Tool: Autovoila (Lossfunk) + Claude Code (claude-sonnet-4-6)",
                  S["CoverMeta"]),
        PageBreak(),
    ]

    # ── SLIDE 1: Research Question ───────────────────────────────────────────
    story += slide_header("1. Starting Research Question")
    story += [
        sec("The Claim"),
        body(
            "Slot-based object-centric representations trained under a <b>hormetic</b> "
            "Information-Bottleneck schedule (beta increasing progressively during training) "
            "retain object identity through occlusion more robustly than identical architectures "
            "trained at fixed beta or non-monotone beta schedules."
        ),
        sp(6),
        sec("The Intervention is Minimal"),
        b("Same architecture, same parameter count, same final beta"),
        b("Only the beta(t) trajectory differs across 6 conditions"),
        b("Primary metric: identity-retention accuracy under occlusion on CLEVRER and ADEPT"),
        sp(6),
        sec("One-Sentence Version"),
        body(
            "<i>A slot-based object representation trained under a hormetic IB schedule maintains "
            "object identity through occlusion more robustly than fixed-beta or non-monotone-beta "
            "baselines, measurable on CLEVRER and ADEPT against published slot-based "
            "persistence systems.</i>"
        ),
        PageBreak(),
    ]

    # ── SLIDE 2: Autoresearch Flow ───────────────────────────────────────────
    story += slide_header("2. Autoresearch Flow")

    col_w = [CW * 0.5 - 6, CW * 0.5 - 6]
    left_col = [
        Paragraph("<b>Tool</b>", S["SlideSection"]),
        body("Autovoila (Lossfunk prompt) on Claude Code (claude-sonnet-4-6), "
             "YOLO mode via cco."),
        sp(8),
        Paragraph("<b>Session Flow</b>", S["SlideSection"]),
        b("Provided the Research Question Sharpener as starting prompt"),
        b("Claude scaffolded the full project in one session: "
          "Slot Attention encoder, VIB head, 6 schedule classes, "
          "training/eval pipeline, data loaders"),
        b("Ran a synthetic pilot (toy Gaussian data, CPU, 100 steps) "
          "to verify the pipeline"),
        b("No real CLEVRER/ADEPT runs — require A100 + ~450 GPU-hours"),
    ]
    right_col = [
        Paragraph("<b>Customization</b>", S["SlideSection"]),
        b("None to the Autovoila prompt"),
        b("Research proposal was detailed enough that Claude needed "
          "very little mid-session direction"),
        b("Outputs reviewed but not edited"),
        sp(8),
        Paragraph("<b>Human Intervention Level: Low</b>", S["SlideSection"]),
        body("Provided initial framing (Research Question Sharpener), "
             "reviewed output, did not touch artifact code or results."),
    ]

    tbl = Table(
        [[left_col, right_col]],
        colWidths=col_w,
    )
    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, -1), 12),
        ("LEFTPADDING", (1, 0), (1, -1), 12),
        ("RIGHTPADDING", (1, 0), (1, -1), 0),
        ("LINEAFTER", (0, 0), (0, -1), 0.5, RULE),
    ]))
    story += [tbl, PageBreak()]

    # ── SLIDE 3: Results ─────────────────────────────────────────────────────
    story += slide_header("3. Summary of Results")
    story += [
        warn("All numbers from synthetic experiments (CPU only). "
             "Not from CLEVRER or ADEPT. 3 seeds x 300 steps; mean shown."),
        sp(4),
    ]

    t_data = [
        ["Condition", "IRA @k=2", "IRA @k=4", "IRA @k=6", "Slot Collapse"],
        ["Hormetic-Sigmoid",      "0.248", "0.254", "0.252", "39.6%"],
        ["Hormetic-Cosine",       "0.291", "0.277", "0.304", "62.6%"],
        ["Linear",                "0.283", "0.237", "0.233", "50.0%"],
        ["Reverse",               "0.335", "0.242", "0.185", "53.7%"],
        ["Random Permutation",    "0.314", "0.281", "0.285", "50.9%"],
        ["Fixed-beta (baseline)", "0.318", "0.227", "0.207", "70.4%"],
    ]
    col_w2 = [CW * f for f in [0.32, 0.17, 0.17, 0.17, 0.17]]
    t = Table(t_data, colWidths=col_w2)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT]),
        ("BACKGROUND",   (0, 1), (0, 2),  HexColor("#e0f2fe")),
        ("GRID",         (0, 0), (-1, -1), 0.3, RULE),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story += [t, sp(6)]
    story += [
        body("<b>Collapse finding:</b> Hormetic-Sigmoid has the lowest collapse (39.6%) "
             "vs Fixed-beta (70.4%). Directionally consistent with the hypothesis mechanism."),
        body("<b>IRA finding:</b> No condition dominates clearly on identity retention. "
             "High variance (std often 50-100% of mean) means 3 seeds is insufficient "
             "for statistically meaningful comparison."),
        PageBreak(),
    ]

    # ── SLIDE 4: What I Learned ──────────────────────────────────────────────
    story += slide_header("4. What I Learned About the Research Question")
    story += [
        sec("What the AI Confirmed"),
        b("The experimental design is correct and the controls are well-chosen"),
        b("The code runs; the question can be operationalized"),
        b("The scaffold is a genuine starting point for real experiments"),
        sp(8),
        sec("What Surprised Me"),
        body(
            "The question is easier to produce <i>results that look like evidence</i> "
            "without being evidence than I expected. The numerical ordering on synthetic data "
            "matches the hypothesis directionally, but that synthetic data was never validated "
            "as a proxy for CLEVRER/ADEPT dynamics. A direction match on toy data is not a result."
        ),
        sp(8),
        sec("The Deeper Gap"),
        body(
            "The research question is well-posed at the proposal level, but the meaningful "
            "empirical content lives entirely in the actual training runs on real data with "
            "real occlusion events. That is precisely what the auto-research system could not do. "
            "The gap between 'scaffold is correct' and 'experiment ran' is the entire scientific "
            "content of this project."
        ),
        PageBreak(),
    ]

    # ── SLIDE 5: Critique — Correct ──────────────────────────────────────────
    story += slide_header("5. Critique: What the AI Got Right")
    story += [
        good("The codebase is structurally clean and correct."),
        sp(4),
    ]

    cr_data = [
        ["Component", "Why it's correct"],
        ["VIB loss",
         "L = recon_loss + beta x KL(q(z|x) || p(z)) with closed-form diagonal Gaussian KL"],
        ["Hungarian matching",
         "Correctly used for slot-to-object assignment in identity-retention evaluation"],
        ["6-condition ablation",
         "Random permutation + reverse baselines are the right controls to isolate schedule geometry"],
        ["Identity-retention metric",
         "Same slot index maps to same object pre/post occlusion at k in {4,8,16,32} frames"],
        ["Data loaders",
         "CLEVRER + ADEPT interfaces structurally correct; would work with real datasets present"],
        ["README",
         "Honestly documents the claim, expected compute (~450 GPU-hours), and data requirements"],
    ]
    cr_col = [CW * 0.28, CW * 0.72]
    cr_t = Table(cr_data, colWidths=cr_col)
    cr_t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  TEAL),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0,1), (-1, -1), [white, LIGHT]),
        ("GRID",         (0, 0), (-1, -1), 0.3, RULE),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [cr_t, PageBreak()]

    # ── SLIDE 6: Critique — Wrong ────────────────────────────────────────────
    story += slide_header("6. Critique: What the Original AI Run Got Wrong")

    left2 = [
        warn("1. The experiments are not the experiments."),
        body("Synthetic Gaussian data is not CLEVRER or ADEPT. "
             "The claim is about visual occlusion in video. "
             "The AI ran what it could and reported it as informative. "
             "It is not."),
        sp(5),
        warn("2. Eval bug: IRA @k=4 was always 0.0."),
        body("Off-by-one in the original evaluation: 'if T > k' with T=4, k=4 "
             "silently skipped the k=4 bucket, returning 0.0. "
             "This was not flagged as anomalous. Fixed by changing to 'if T >= k' "
             "and T=6 to match config."),
        sp(5),
        warn("3. 75% slot collapse with no diversity penalty."),
        body("The original run had no mechanism to prevent slot collapse. "
             "When slots degenerate, IRA measures noise routing. "
             "Fixed by adding a slot diversity auxiliary loss (lambda=0.05)."),
    ]
    right2 = [
        warn("4. Single seed, 100 steps: not enough signal."),
        body("One seed at 100 steps on toy data gives zero statistical power. "
             "The IRA ordering at k=2 (Hormetic-Cosine first) was likely noise. "
             "Fixed by running 3 seeds at 300 steps with aggregated mean +/- std."),
        sp(5),
        warn("5. The paper was never written."),
        body("The caisc_2026.tex in the original repo is the blank template. "
             "No abstract, no methods, no results in paper form. "
             "A full CAISc 2026 draft has now been written."),
        sp(5),
        warn("6. No IB-plane trajectories, no OOD evaluation."),
        body("Both listed as secondary measurements in the design. "
             "Neither was produced, even on the synthetic dataset."),
    ]

    tbl2 = Table(
        [[left2, right2]],
        colWidths=[CW * 0.5 - 6, CW * 0.5 - 6],
    )
    tbl2.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (0, -1),  0),
        ("RIGHTPADDING", (0, 0), (0, -1),  12),
        ("LEFTPADDING",  (1, 0), (1, -1),  12),
        ("RIGHTPADDING", (1, 0), (1, -1),  0),
        ("LINEAFTER",    (0, 0), (0, -1),  0.5, RULE),
    ]))
    story += [tbl2, PageBreak()]

    # ── SLIDE 7: Reflection ──────────────────────────────────────────────────
    story += slide_header("7. Reflection: Limits of Auto-Research on This Question")

    left3 = [
        good("What the AI did well"),
        b("Translated a well-specified proposal into working code fast; "
          "a human needs 2-4 weeks for the same scaffold"),
        b("Picked the right components: VIB, Hungarian matching, "
          "SAVi-style temporal slot attention, spatial broadcast decoder"),
        b("Designed the correct ablation structure including "
          "matched-beta controls"),
        sp(10),
        sec("What kind of question would AI do better on?"),
        body("A question where compute is small, data is accessible, "
             "evaluation is unambiguous, and prior literature has a "
             "clear quantitative baseline to beat. Something like: "
             "'Does this architectural change improve ARI on CLEVRER "
             "in under 10 GPU-hours?' That is nearly fully automatable. "
             "This question requires 450 GPU-hours, two datasets, "
             "and evaluation that depends on ruling out multiple confounds."),
    ]
    right3 = [
        warn("What the AI failed at"),
        b("<b>Cannot run experiments requiring GPUs and datasets "
          "it does not have.</b> The entire empirical content of "
          "the claim lives in 450 GPU-hours on CLEVRER/ADEPT."),
        b("<b>Did not recognize when its own results were invalid</b> "
          "in the first pass. After prompting, diagnosed the eval bug "
          "(k=4 always 0.0), identified slot collapse as the root cause, "
          "and corrected both in a second run."),
        b("<b>'The pipeline ran' is not 'the experiment worked.'</b> "
          "Running without error on toy data is not evidence."),
        b("<b>Cannot construct the interpretive argument.</b> "
          "The novel claim requires reading IB theory, Slot Attention, "
          "and Saxe et al.'s critique together with nuance "
          "the AI cannot provide."),
    ]

    tbl3 = Table(
        [[left3, right3]],
        colWidths=[CW * 0.5 - 6, CW * 0.5 - 6],
    )
    tbl3.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (0, -1),  0),
        ("RIGHTPADDING", (0, 0), (0, -1),  12),
        ("LEFTPADDING",  (1, 0), (1, -1),  12),
        ("RIGHTPADDING", (1, 0), (1, -1),  0),
        ("LINEAFTER",    (0, 0), (0, -1),  0.5, RULE),
    ]))
    story += [tbl3, PageBreak()]

    # ── SLIDE 8: Revised Plan ────────────────────────────────────────────────
    story += slide_header("8. Revised Research Plan")

    left4 = [
        sec("What Stays the Same"),
        b("Core claim is still worth testing; experimental design is correct"),
        b("AI-generated codebase is a usable starting point"),
        sp(10),
        sec("What Was Fixed in This Session"),
        b("Eval bug found and fixed: IRA @k=4 now non-zero across all conditions"),
        b("Slot diversity loss added (lambda=0.05): hormetic_sigmoid collapse "
          "drops from 75% to 39.6%; fixed_beta remains at 70.4%"),
        b("3-seed 300-step ablation run: results have error bars, "
          "not just point estimates"),
        b("Full CAISc 2026 paper draft written with real numbers filled in"),
        sp(8),
        sec("What Still Cannot Be Done Without GPU + Data"),
        b("Experiments on CLEVRER or ADEPT (approx 450 GPU-hours on A100)"),
        b("ARI evaluation on real video frames"),
        b("IB information-plane trajectories at scale"),
    ]
    right4 = [
        sec("What the Corrected Results Show"),
        body(
            "The collapse finding is directionally consistent: "
            "hormetic_sigmoid has the lowest collapse (39.6%) and "
            "fixed_beta has the highest (70.4%). "
            "This supports the mechanism hypothesis that progressive "
            "compression prevents early slot degeneration."
        ),
        sp(8),
        sec("What the Corrected Results Do Not Show"),
        body(
            "The IRA numbers do not cleanly favor hormetic conditions. "
            "No single condition dominates across k=2,4,6. "
            "Variance is high (std often 50-100% of mean). "
            "3 seeds on CPU at 300 steps is insufficient for "
            "statistically meaningful schedule comparison. "
            "The collapse finding is the only signal worth reporting."
        ),
    ]

    tbl4 = Table(
        [[left4, right4]],
        colWidths=[CW * 0.5 - 6, CW * 0.5 - 6],
    )
    tbl4.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (0, -1),  0),
        ("RIGHTPADDING", (0, 0), (0, -1),  12),
        ("LEFTPADDING",  (1, 0), (1, -1),  12),
        ("RIGHTPADDING", (1, 0), (1, -1),  0),
        ("LINEAFTER",    (0, 0), (0, -1),  0.5, RULE),
    ]))
    story += [tbl4, PageBreak()]

    # ── SLIDE 9: Summary ─────────────────────────────────────────────────────
    story += slide_header("9. Summary")

    sum_data = [
        ["Item",                        "Status"],
        ["Research question",           "Well-posed, falsifiable, still worth testing"],
        ["Codebase / scaffold",         "AI-generated, structurally correct, usable"],
        ["Experiments (CLEVRER/ADEPT)", "Not run — require GPU and real datasets"],
        ["Synthetic pilot",             "Corrected: eval bug fixed, 3-seed 300-step ablation, diversity loss added"],
        ["Paper draft",                 "Written: CAISc 2026 draft with corrected numbers"],
        ["GitHub repo",                 "github.com/Incharajayaram/hormetic-ib-slot"],
    ]
    sum_col = [CW * 0.32, CW * 0.68]
    sum_t = Table(sum_data, colWidths=sum_col)
    sum_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  10),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [white, LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.3, RULE),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))

    story += [
        sum_t,
        sp(16),
        HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=12),
        Paragraph(
            "<b>The honest one-sentence version:</b>",
            S["ClosingBig"],
        ),
        Paragraph(
            "The AI built the lab but could not run the experiment, "
            "and did not recognize that the dry-run it did run had failed.",
            S["ClosingBody"],
        ),
        sp(10),
        Paragraph(
            "Generated with Autovoila (Lossfunk) + Claude Code.   "
            "Human contribution: research proposal, Research Question Sharpener, this critique.",
            S["Caption"],
        ),
    ]

    doc.build(story)
    print(f"Deck written to: {out_path}")


if __name__ == "__main__":
    build_deck(
        "/home/incharanew/autovoila/all-spikes/hormetic-ib-slot/lossfunk_submission_deck.pdf"
    )
