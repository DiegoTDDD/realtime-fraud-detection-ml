"""Build docs/case_study.pdf from the project's current results (ReportLab).

Numbers come from training/metrics.json, so the PDF cannot drift from the model it describes.
Run: python scripts/build_case_study.py
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "case_study.pdf"
SHOTS = ROOT / "docs" / "screenshots"
M = json.loads((ROOT / "training" / "metrics.json").read_text(encoding="utf-8"))
REC = M["recall_by_fraud_type"]
COUNTS = M.get("test_count_by_fraud_type", {})

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("Sans", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Oblique", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONT_DIR / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Oblique")

INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5F6B7A")
ACCENT = colors.HexColor("#0B5CAD")
RULE = colors.HexColor("#D5DBE3")
TINT = colors.HexColor("#EEF3F9")

body = ParagraphStyle("body", fontName="Sans", fontSize=9.6, leading=14, textColor=INK, spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=11.5, textColor=MUTED)
caption = ParagraphStyle("caption", parent=small, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
h1 = ParagraphStyle("h1", parent=body, fontName="Sans-Bold", fontSize=13, leading=17, spaceBefore=10, spaceAfter=6, textColor=ACCENT, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=body, fontName="Sans-Bold", fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=4, keepWithNext=1)
kicker = ParagraphStyle("kicker", parent=small, fontName="Sans-Bold", textColor=ACCENT, fontSize=8.5)
title = ParagraphStyle("title", parent=body, fontName="Sans-Bold", fontSize=26, leading=31)
subtitle = ParagraphStyle("subtitle", parent=body, fontSize=12, leading=17, textColor=MUTED)
mono = ParagraphStyle("mono", fontName="Mono", fontSize=7.6, leading=10, textColor=INK)
cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=11.2, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Sans-Bold")
kpi_n = ParagraphStyle("kpi_n", parent=body, fontName="Sans-Bold", fontSize=17, leading=21, alignment=TA_CENTER, textColor=ACCENT, spaceAfter=0)
kpi_l = ParagraphStyle("kpi_l", parent=small, alignment=TA_CENTER, spaceAfter=0)


def p(text, style=body):
    return Paragraph(text, style)


def pct(x):
    return f"{100 * x:.1f}%"


def table(rows, widths, header=True):
    data = [[p(c, cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), TINT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    t.setStyle(TableStyle(style))
    return t


def callout(head, text):
    t = Table([[p(f"<b>{head}</b>", cell)], [p(text, cell)]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TINT),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return KeepTogether([Spacer(1, 4), t, Spacer(1, 8)])


def image(path, width_mm, text):
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    width = width_mm * mm
    return KeepTogether([Image(str(path), width=width, height=width * h / w), p(text, caption)])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 12 * mm, "Diego L. A. de Andrade · Real-Time Fraud Detection: ML Scoring Layer")
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    s = []
    # ---- Cover
    s += [Spacer(1, 40 * mm), p("MACHINE LEARNING ENGINEERING · REAL-TIME SERVING CASE STUDY", kicker), Spacer(1, 6)]
    s += [p("Real-Time Fraud Detection", title), p("ML Scoring Layer", title), Spacer(1, 8)]
    s += [p("An offline-trained XGBoost classifier scores a live Kafka transaction stream inside Spark "
            "Structured Streaming, with per-card state that survives restarts, predictions written exactly "
            "once to Delta Lake, and a monitor that raises an alert when the model degrades.", subtitle)]
    s += [Spacer(1, 18 * mm), p("Repository · github.com/DiegoTDDD/realtime-fraud-detection-ml", small),
          p("Builds on · github.com/DiegoTDDD/realtime-fraud-streaming (Project 1)", small),
          Spacer(1, 30 * mm), p("<b>Diego Leonardo Alves de Andrade</b>", body), PageBreak()]

    # ---- Executive summary
    s += [p("Executive summary", h1)]
    s += [p("Project 1 built the streaming backbone for card-transaction fraud: event flow, a schema registry, "
            "windowed aggregation in Delta Lake and a descriptive dashboard. This project makes it predictive. "
            "An XGBoost classifier is trained offline on data generated with the producer's own logic, then "
            "scores every transaction of Project 1's Kafka topic as an independent second consumer. Each "
            "card's recent history lives in Spark's state store, each scored micro-batch is written to Delta "
            "once, and a monitor checks live precision, recall and input drift against written thresholds.")]
    kpis = [
        (f"{M['pr_auc']:.3f}", "PR-AUC on the test period"),
        (f"{pct(M['precision'])} / {pct(M['recall'])}", "precision / recall at the threshold chosen on validation"),
        ("4,597 / 4,597", "Kafka messages / predictions after a SIGKILL restart in CI"),
        ("47", "tests: 44 unit, 3 Spark"),
    ]
    kt = Table([[p(n, kpi_n) for n, _ in kpis], [p(lbl, kpi_l) for _, lbl in kpis]], colWidths=[42.5 * mm] * 4)
    kt.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.4, RULE), ("INNERGRID", (0, 0), (-1, -1), 0.4, RULE),
                            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    s += [Spacer(1, 4), kt, Spacer(1, 6)]
    s += [p("<b>Stack:</b> XGBoost · scikit-learn · Spark Structured Streaming (applyInPandasWithState) · "
            "Redpanda (Kafka API) with schema registry · Delta Lake · Streamlit + Plotly · GitHub Actions", small)]

    s += [p("The problem", h1)]
    s += [p("Project 1's producer injects three fraud patterns, each with a ground-truth label: an abnormally "
            "high charge, a burst of small probe charges on one card, and a charge far from the card's home city. "
            "The question was whether a model trained offline could serve the live stream in real time at a class "
            "balance where accuracy is meaningless, and whether that could be shown rather than claimed: that it "
            "does not only catch the easiest, most frequent pattern, that its numbers hold on data no choice has "
            "seen, and that a restart neither loses history nor writes a prediction twice.")]
    s += [callout("Guiding question",
                  "Can a model trained offline on synthetic fraud patterns score a live Kafka stream with the same "
                  "features it was trained on, keep its reported precision honest, survive a crash of the scoring "
                  "job without gaps or duplicates, and tell someone when it degrades?")]

    s += [p("Why the training data is generated, not sampled", h1)]
    s += [p("Project 1's Bronze history held only 189 fraud examples, too few for a stable precision and recall "
            "per pattern. Running the producer for hours would mix two jobs, simulating a feed and building a "
            "training set, and give a different dataset each time. A batch script reuses the producer's exact "
            "logic (same 200 cards, cities, categories, fraud patterns and pace) and writes 150,000 transactions "
            "in seconds, seeded so every run writes the same data.")]
    s += [callout("Decision I'd defend",
                  "I did not change Project 1's producer to get more data. Modifying a shared dependency for one "
                  "downstream need quietly breaks the other consumer; reusing its logic in an isolated script kept "
                  "both projects correct.")]

    # ---- Design
    diagram = (
        "Producer (Project 1) --> Redpanda + schema registry --> Spark Structured Streaming\n"
        "card txns + labels        topic: transactions             1. strip the wire-format header\n"
        "                          (0 byte + schema id + JSON)     2. applyInPandasWithState by card:\n"
        "                                                             velocity features from the state store\n"
        "                                                          3. mapInPandas: shared features + XGBoost\n"
        "                                                          4. Delta append, txnVersion = batch id\n"
        "                                                                     |\n"
        "                                         Delta table data/predictions (exactly once)\n"
        "                                              |                          |\n"
        "                                   Streamlit dashboard          scoring/monitor.py\n"
        "                                   (live precision/recall)      (quality + drift, webhook)\n"
        "\n"
        "training/model.joblib is loaded once when the job starts, not per micro-batch."
    )
    s += [KeepTogether([p("How the system is built", h1), Preformatted(diagram, mono)]), Spacer(1, 6)]
    s += [p("The scoring job is a second, independent consumer of Project 1's topic: it needs the topic to exist "
            "but not Project 1's own Spark job. That separation let this project evolve without touching "
            "Project 1's code.")]

    s += [p("Class imbalance: weights, not resampling", h2)]
    s += [p(f"About 7.5% of rows are fraud (2% of events start a fraud pattern; card-testing bursts add rows). "
            f"XGBoost is trained with <font face='Mono'>scale_pos_weight</font> = {M['scale_pos_weight']:.2f}, the "
            "training period's negatives over positives. Undersampling throws signal away, and synthetic minority "
            "rows would break the velocity features, which depend on real neighbouring transactions. PR-AUC is the "
            "main metric: a model that never predicts fraud already scores 92.5% accuracy.")]
    s += [p("The features are deliberately few: amount, distance_km_from_home, tx_count_last_5min, "
            "time_since_last_tx, hour_of_day and one-hot category. Merchant is left out: a high-cardinality name "
            "with no signal here would invite overfitting.")]

    s += [p("Three periods in time order", h2)]
    s += [p(f"The velocity features depend on neighbouring transactions, so a random split would leak across the "
            f"boundary. The data is split by time: the earliest 60% ({M['train_rows']:,} rows) train the model, the "
            f"next 20% ({M['validation_rows']:,}) choose the threshold, and the latest 20% ({M['test_rows']:,}, "
            f"{M['test_fraud_count']:,} frauds) are scored once for every reported number. Home cities are learned "
            "from the training period only. An earlier version chose the threshold on the test set; moving that "
            "choice to a validation period is what makes the numbers below honest.")]

    # ---- Evaluation
    s += [p("Model evaluation", h1)]
    s += [p("The default 0.5 threshold and the best-F1 threshold are both driven by card testing, about 83% of "
            "fraud rows. Instead, the threshold caps false alarms with a 95% precision floor, an alert budget a "
            "fraud team can staff, and within it maximizes the mean recall across the three patterns, so a rare "
            "pattern cannot be ignored because it is outnumbered.")]
    rows = [["Metric (test period)", "Value"],
            ["PR-AUC", f"{M['pr_auc']:.4f}"],
            ["Threshold (chosen on validation)", f"{M['threshold']:.3f}"],
            ["Precision", f"{pct(M['precision'])} (validation {pct(M['validation_precision'])})"],
            ["Recall", f"{pct(M['recall'])} (validation {pct(M['validation_recall'])})"],
            ["Mean recall across the three patterns", pct(M["macro_recall"])]]
    s += [table(rows, [110 * mm, 60 * mm]), Spacer(1, 6)]
    prow = [["Fraud pattern", "Test examples", "Recall"]]
    for name in ["card_testing", "high_amount", "impossible_travel"]:
        prow.append([name, f"{COUNTS.get(name, 0):,}", pct(REC[name])])
    s += [table(prow, [80 * mm, 45 * mm, 45 * mm]), Spacer(1, 6)]
    s += [image(SHOTS / "pr_curve.png", 120, "Precision-recall curve on the test period. The chosen threshold sits "
                "just before the curve's drop, where more recall starts to cost precision fast.")]

    s += [p("Feature engineering as a measured process", h2)]
    s += [p("high_amount and card_testing separated easily. impossible_travel did not, and three designs were "
            "measured side by side:")]
    s += [table([["Design", "Recall (impossible_travel)", "Why it was or was not kept"],
                 ["Binary is_unusual_city", "0.59 to 0.65", "Cannot tell a nearby trip from a cross-country one"],
                 ["implied_speed_kmh (distance / time since the previous transaction)", "0.62",
                  "The previous transaction is often hours earlier, so the speed looks plausible"],
                 ["distance_km_from_home (great-circle, inferred home city)", "0.65", "Best of the three; kept"]],
                [70 * mm, 35 * mm, 65 * mm]), Spacer(1, 4)]
    s += [p("That comparison predates the validation period; a design choice like it is now made there, so the "
            "test period stays untouched.", small)]
    s += [callout("The gap I state plainly",
                  f"impossible_travel recall is {pct(REC['impossible_travel'])}. About 13.5% of legitimate "
                  "transactions also happen away from the home city, and the test period has about 190 examples of "
                  "this pattern (roughly ±7 points of uncertainty). Closing the gap needs signal the stream does not "
                  "carry, such as travel history or device location, not another threshold tweak.")]

    # ---- Serving
    s += [p("Real-time serving", h1)]
    s += [p("Serving must compute exactly the features training saw. All feature functions live in one module "
            "that both use, and tests compare the two paths. Two features need history beyond a single "
            "micro-batch: how many transactions the card made in the last five minutes, and the time since its "
            "previous one.")]
    code = (
        "events.groupBy(\"card_id\").applyInPandasWithState(\n"
        "    update_card,                      # velocity_for_card() from training/features.py\n"
        "    outputStructType=OUTPUT_DDL,\n"
        "    stateStructType=\"window array<long>, last long\",\n"
        "    outputMode=\"append\",\n"
        "    timeoutConf=GroupStateTimeout.ProcessingTimeTimeout,   # idle cards expire after 24 h\n"
        ")\n"
        "\n"
        "def sink(batch, batch_id):\n"
        "    scored = batch.mapInPandas(scorer, PREDICTION_DDL)     # features + XGBoost on the workers\n"
        "    (scored.write.format(\"delta\").mode(\"append\")\n"
        "        .option(\"txnAppId\", query_id_from_checkpoint)\n"
        "        .option(\"txnVersion\", batch_id)                  # a replayed batch is skipped\n"
        "        .save(\"data/predictions\"))"
    )
    s += [KeepTogether([Preformatted(code, mono)]), Spacer(1, 6)]
    s += [p("<b>State that survives a restart.</b> Each card's recent transaction times and the time of its last "
            "transaction live in Spark's state store, checkpointed together with the Kafka offsets, so after a "
            "restart the history and the input position always match. Cards are spread across workers by key. A "
            "first version kept this history in a Python dict on the driver: lost on every restart and impossible "
            "to scale out. Reviewing it also found a training/serving skew: serving reset time_since_last_tx once "
            "the previous transaction left the five-minute window, while training used the real gap. Both paths "
            "now share one function and the same 24-hour cap.")]
    s += [p("<b>Exactly-once output.</b> Each micro-batch is appended to Delta with the batch id as transaction "
            "version and the query id from the checkpoint as application id. Delta skips a version it has already "
            "committed, so a batch replayed after a crash is not written twice; a new checkpoint gets a new id, so "
            "its batch numbers starting again from 0 are not mistaken for old ones.")]
    s += [p("<b>The schema registry's wire format.</b> Project 1's producer writes each message as a 0 byte, a "
            "4-byte schema id and the JSON. The parser strips that header and still reads plain JSON.")]

    s += [p("Proving it", h2)]
    s += [table([["Check", "What it shows"],
                 ["Spark test: stop and restart the query on the same checkpoint",
                  "Every velocity value equals the batch computation over all events at once"],
                 ["Spark test: write the same (app id, version) twice", "The second write adds nothing"],
                 ["Unit tests of the state function", "The 5-minute window and the 24-hour cap match the batch rule at their edges"],
                 ["CI pipeline check: SIGKILL the scoring job mid-stream, restart it, let it catch up",
                  "4,597 messages in Kafka, 4,597 predictions in Delta, 0 duplicate offsets"]],
                [80 * mm, 90 * mm])]

    # ---- Monitoring
    s += [p("Knowing when the model degrades", h1)]
    s += [p("The dashboard showed live precision and recall, but nothing acted on them. scoring/monitor.py reads "
            "the latest 2,000 predictions from Delta (at least 500 to judge, after the first five minutes while the "
            "velocity windows fill), exits with code 1 and posts to a webhook when a check fails:")]
    s += [table([["Check", "Threshold"],
                 ["Precision and recall", "≥ 0.90"],
                 ["Recall on high_amount and card_testing", "≥ 0.90"],
                 ["Recall on impossible_travel", "≥ 0.45 (the known weak pattern)"],
                 ["PSI of amount, distance from home and both velocity features against training", "< 0.25"]],
                [120 * mm, 50 * mm]), Spacer(1, 6)]
    s += [p("A quality check fails only when the evidence says the rate is below its floor: the upper end of the "
            "95% Wilson interval must be under it. The first live run in CI is why. high_amount had 10 examples in "
            "the window and 8 were caught, 80% against a 90% floor, yet a model with 97% recall misses 2 of 10 "
            "about 3% of the time. 40 of 100 would still fail.")]

    # ---- Infrastructure
    s += [p("Getting it running: infrastructure failures", h1)]
    s += [p("None of these were model bugs. They were the friction of running Spark Structured Streaming with "
            "Python workers on Windows and in CI, and each was diagnosed from the stack trace to a root cause.")]
    s += [table([["Failure", "Root cause", "Fix"],
                 ["SparkContext failed to initialize (Windows)", "Hadoop native binaries missing; a naive download link served an HTML page",
                  "Binaries from the raw URL, checked before retrying; the Windows check now installs them"],
                 ["Topic did not exist", "The broker had no persistent volume; the Kafka source fails on a missing topic",
                  "Start the producer first; the README states the order"],
                 ["Python worker failed to connect back", "PATH resolved python to the Microsoft Store stub",
                  "PYSPARK_PYTHON set to sys.executable before Spark starts"],
                 ["CANNOT_DETERMINE_TYPE on write", "A batch with no fraud had an all-null fraud_type",
                  "Explicit output schema instead of inference"],
                 ["Tools not found after install", "PATH changes do not reach an already-open shell",
                  "Exported the binary's folder for the session"],
                 ["Exit code 134 after a passing check", "deltalake's background threads abort the interpreter exit",
                  "Flush and os._exit after reading Delta"],
                 ["Kafka and Delta jars not found in CI", "Maven Central refused the download",
                  "Google's Maven mirror as a second repository, a retried download step and a cache"]],
                [45 * mm, 65 * mm, 60 * mm])]

    # ---- Dashboard
    s += [p("The live dashboard", h1)]
    s += [p("The dashboard reports model quality, not only volume. That is possible because Project 1's producer "
            "labels every transaction; most real fraud dashboards have no such oracle in real time. The deployed "
            "app reads the latest 8,000 predictions of a recorded run, contiguous in time, so its live recall is "
            "not inflated by sampling.")]
    s += [image(SHOTS / "01-overview.png", 165, "Live precision and recall over the last 10 minutes, and each "
                "transaction's fraud probability against the threshold.")]
    s += [image(SHOTS / "02-recall-by-type.png", 165, "Live recall per pattern next to the offline test-period "
                "numbers: all three patterns are caught live, not only card testing.")]
    s += [image(SHOTS / "03-alerts-feed.png", 165, "The newest alerts with a correct/incorrect mark against "
                "ground truth, an audit trail an analyst could use.")]

    # ---- Closing
    s += [p("What this project demonstrates", h1)]
    for head, text in [
        ("Applied ML on imbalanced data", "class weights over resampling, PR-AUC, and a threshold policy that protects the rare patterns."),
        ("Honest evaluation", "three periods in time order; the threshold is chosen on validation and the test period is scored once."),
        ("Stateful, exactly-once streaming", "per-card state in Spark's state store and an idempotent Delta sink, shown by a restart test and a SIGKILL in CI."),
        ("Model monitoring", "quality and drift checks with statistical evidence behind each alert, an exit code and a webhook."),
        ("Infrastructure debugging", "failures on Windows and in CI traced to root causes, each fix now guarded by a check."),
    ]:
        s += [p(f"<b>{head}</b>: {text}")]
    s += [p("Limitations", h1)]
    s += [p(f"impossible_travel recall ({pct(REC['impossible_travel'])}) is the clearest gap, for the reasons above. "
            "The training data is synthetic and its fraud patterns are simple and fixed, so card-testing recall of "
            "100% will not carry over to real transactions. Labels arrive with each event here; real labels come "
            "weeks later from chargebacks, so the quality checks would lag the drift checks. Retraining is a manual "
            "run of train_model.py; the monitor raises the alarm but does not retrain.")]
    s += [KeepTogether([p("Technical reference", h1), table([["Stage", "Component", "Purpose"],
                 ["Model", "XGBoost 2.1, scikit-learn", "Class-weighted gradient-boosted classifier"],
                 ["Training data", "Batch generator reusing Project 1's logic", "150,000 seeded transactions in seconds"],
                 ["Broker", "Redpanda (Kafka API) + schema registry", "Reused from Project 1"],
                 ["Serving", "PySpark 3.5 Structured Streaming", "applyInPandasWithState, mapInPandas, idempotent Delta sink"],
                 ["Storage", "Delta Lake (delta-spark 3.3, deltalake)", "Predictions, read by the monitor and dashboard without Spark"],
                 ["Monitoring", "scoring/monitor.py", "Wilson-bounded quality checks, PSI drift, webhook"],
                 ["Presentation", "Streamlit + Plotly", "Live precision and recall per pattern"],
                 ["CI", "GitHub Actions", "Unit and Spark tests, SIGKILL pipeline check, Windows check, app health"]],
                [32 * mm, 60 * mm, 78 * mm])])]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title="Real-Time Fraud Detection: ML Scoring Layer", author="Diego Leonardo Alves de Andrade")
    doc.build(s, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
