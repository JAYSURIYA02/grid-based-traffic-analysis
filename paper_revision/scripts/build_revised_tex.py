"""Builds paper_revision/revised_paper.tex from paper/paper_final.tex by explicit,
asserted text replacements (the original file is only read)."""
import os, re
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
s = open(os.path.join(R, "paper", "paper_final.tex"), encoding="utf-8").read().replace("\r\n", "\n")

def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:80])
    s = s.replace(old, new)

def rep_between(start, end, new, keep_end=True):
    """replace text from start marker (inclusive) to end marker (exclusive if keep_end)"""
    global s
    i = s.index(start); j = s.index(end, i)
    s = s[:i] + new + (s[j:] if keep_end else s[j + len(end):])

# ---------------- title ----------------
rep_between("\\title{", "}\n\n\\author{", "\\title{Evaluating Lightweight Grid-Based Vehicle Classification and Counting\nUsing Resolution Sensitivity Analysis and Sequential--Parallel Benchmarking", keep_end=True)

# ---------------- abstract ----------------
rep_between("\\begin{abstract}\n", "\\end{abstract}", r"""\begin{abstract}
Grid-based (``virtual lattice'') motion detection processes traffic video
on commodity CPUs without training, but the published lattice detector
stops at a binary occupancy grid and reports no accuracy. We extend its
front end with geometric vehicle classification, tracking and
line-crossing counting (plus density estimation and a speed estimator
whose accuracy is not evaluated) in a sequential and a two-lane parallel
Python implementation, and study the pipeline experimentally on UA-DETRAC.
A cross-implementation check found two preprocessing divergences; after
fixing them, both implementations give identical results on 12 benchmark
videos. A four-workload benchmark on a 4-core/8-thread CPU separates lane
count from threading: the thread pools slowed the same two-lane workload
(time ratio $0.860\pm0.077$, slower on all 12 videos), and the
$1.26\times$ higher ROI-frame throughput of the two-lane implementation
comes from sharing decoding and preprocessing between lanes. A
grid-resolution study with site-grouped cross-validation, counting ground
truth derived from annotated trajectories and 44 videos from unseen sites
shows that, once thresholds are re-tuned per grid, no resolution changes
macro-F1 significantly, whereas $10{\times}12$ raises the per-video count
error from 16.0 to 23.1 vehicles (Holm-adjusted $p=0.010$). The tuned
classifier exceeds an always-Car baseline in macro-F1 but not in
weighted-F1, and the counter registers only downward crossings, so it
misses 44\% of the test crossings, most of them in Lane~2.
""")

# ---------------- intro: contributions ----------------
rep_between("We contribute: (1)", "\\begin{table}[t]\n\\centering\n\\caption{Baseline", r"""This paper is an experimental study of such a lightweight grid-based
pipeline rather than a proposal of a state-of-the-art classifier. We
contribute: (1) a classification, tracking, counting and
adaptive-density layer on that front end, implemented identically in a
sequential (\code{seq.py}) and a two-lane parallel
(\code{rparallel\_he3.py}) implementation, plus a calibration-based speed
estimator that is implemented but not validated; (2) a parallel design
with persistent pools, per-lane \code{LaneTracker} state and a
compute/draw split; (3) a cross-implementation check that found and fixed
two preprocessing divergences; (4) a four-workload benchmark that
separates the effect of processing two lanes from the effect of
threading; and (5) a grid-resolution study with site-grouped
cross-validation, trajectory-based counting ground truth, paired
statistical tests and an unseen-site test set.

""")

# ---------------- speed subsection: remove figure, add details ----------------
rep_between("\\subsection{Speed Estimation (Implemented, Not Validated)}", "\\subsection{Parallel Implementation}", r"""\subsection{Speed Estimation (Implemented, Not Validated)}
\label{sec:speed}
The image-to-metric scale is calibrated in the web interface from two
clicked points that span a known real-world distance (a lane width, a
lane-mark period or a vehicle width): with pixel distance $d$ between the
points (converted from display to native video coordinates) and real
distance $L$, $k=d/L$ pixels per meter. A command-line tool can average
several pairs and warns when their coefficient of variation exceeds 15\%.
For each track, the Euclidean centroid displacement since the last speed
sample is accumulated until it reaches three cell heights, which
suppresses grid quantization; a sample is then
\begin{equation}
v=\frac{\Delta d_{\text{px}}/k}{\Delta n/f}\times3.6\ \text{km/h},
\label{eq:speed}
\end{equation}
where $\Delta n$ frames have elapsed at the video frame rate $f$. The
displayed speed is the mean of the track's last 15 samples. The estimate
is a magnitude (direction is ignored), there is no outlier rejection
beyond the displacement gate, and a single scale $k$ ignores perspective
foreshortening. \textbf{Speed accuracy is not evaluated}: no speed ground
truth exists for the videos, and $k=0$ in all benchmark and grid
experiments, which disables the computation. In a qualitative calibrated
run (6\,m reference, 8.721\,px/m; screenshots in the supplementary
material), the overlay showed plausible values (42--43\,km/h) next to
implausible ones (above 400\,km/h), as expected when cluster centroids
jump between frames; validation against measured speeds and
perspective-correct scaling are required \cite{kamil2024,sangsuwan2024}.

""")

# ---------------- counting description ----------------
rep("""is suppressed. A crossing is counted when the centroid passes the ROI's
mid-line (track age $\\geq 2$), unless it lies within 30 frames and 60\\,px
of a previous crossing.""", """is suppressed. A crossing is counted when a track's centroid moves from
above to on or below the ROI's mid-line (track age $\\geq 2$), unless it
lies within 30 frames and 60\\,px of a previous crossing in the same lane.
Only this downward direction is counted, in both ROIs, and there is no
de-duplication between the two ROIs.""")

# ---------------- evaluation subsection: add counting GT + statistics ----------------
rep("""not masked, and matching is greedy; these choices affect both implementations
equally.""", r"""not masked, and matching is greedy; these choices affect both implementations
equally.

\textbf{Counting ground truth.} The number of annotated vehicles is not a
count of line crossings, so crossing events are derived from the
annotated trajectories: for each UA-DETRAC target, the centre of its box
in consecutive annotated frames; an event is recorded when the centre
crosses the counting line ($y=272$), with direction \emph{down}
(previous $y<272\le$ current $y$) or \emph{up}, and is assigned to the
ROI whose $x$-range contains the centre (events in the 12\,px gap between
the ROIs are excluded). Predicted events are the counter's own events,
replayed from per-frame logs; the replay reproduces the reported per-lane
totals in every run. Predicted events inside UA-DETRAC ignored regions
are excluded. A predicted and a ground-truth event match if
$|\Delta t|\le25$ frames (1\,s) and $|\Delta x|\le80$\,px and the
ground-truth event is downward; matching is one-to-one and greedy by
$|\Delta t|/25+|\Delta x|/80$, with a bonus when the counted track's box
overlaps the same annotated vehicle. An unmatched prediction is a
\emph{duplicate} if its vehicle, or a compatible ground-truth event, was
already matched, and a \emph{false count} otherwise; unmatched downward
events are \emph{missed counts}, and upward events are \emph{directional
omissions}. In the combined score both ROIs compete for one list of
ground-truth events, so a vehicle is never credited twice. We report
event precision $TP/P$, recall $TP/G$, $F_1=2TP/(P+G)$ and, per video,
the count error $P-G$ as MAE, RMSE and MAPE (videos with $G>0$).

\textbf{Statistics.} Grid comparisons on the test set are paired (all grids
run on the same videos). The resampling unit is the camera site (12
groups); we report 95\% percentile bootstrap intervals (4000 resamples)
and two-sided paired permutation tests that swap the two grids' results
within randomly chosen sites. For each metric, the three comparisons
against $6{\times}6$ are Holm-adjusted; we also report the correction
over all 18 tests.""")

# ---------------- experimental setup: grid study ----------------
rep("""are computed by \\code{benchmarking/orchestrate\\_benchmark.py}.""", r"""are computed by \code{benchmarking/orchestrate\_benchmark.py}.

\textbf{Grid-resolution study.} Six resolutions ($4{\times}6$ to
$10{\times}12$, rows$\times$columns) were run on workload~C (both lanes,
no threads) on a separate 2-vCPU Linux VM (OpenCV~5.0); its frame rates
are not comparable with the benchmark machine. UA-DETRAC has no
camera-site labels, so sites were inferred from median-background
thumbnails of all 100 videos (road layout, medians, fences, buildings
and street furniture), cross-checked by an automatic gradient-correlation
clustering; doubtful pairs were merged, because over-merging only reduces
the number of folds whereas under-merging leaks scene information. The
26 annotated videos of the four weather folders that are not benchmark
videos (30{,}427 frames, 10 site groups) were used for selection only.
The test set is 44 further videos (67{,}206 frames, 12 site groups) that
no earlier experiment used and whose scenes resemble none of the 38
tuning and benchmark videos; 18 resembling videos were excluded. The
12 benchmark videos share sites with the tuning videos (8 of 12) and had
informed an exploratory analysis, so they are not used for grid
comparisons. The complete video-to-site mapping, with the uncertain
assignments, is supplementary material.

Selection used leave-one-site-out cross-validation on the 26 videos. In
each fold and for each grid $g$, the thresholds $\theta_g$ were chosen
from 810 candidate rules (Bus/Van span thresholds, optionally a Van
aspect limit) to maximise pooled macro-F1 on the training sites. Grids
were then scored by
\begin{equation}
\begin{split}
J(g)={}&0.30\,F_1^{\text{mac}}+0.10\,\min_c F_1^{c}+0.25\,F_1^{\text{cnt}}\\
&+0.15\,R^{\text{det}}+0.10\,(1-\phi)+0.10\,(1-u),
\end{split}
\label{eq:objective}
\end{equation}
where all terms lie in $[0,1]$, are pooled over the scored videos and are
not further normalized: $F_1^{\text{mac}}$ and $F_1^{c}$ are vehicle-level
macro and per-class F1 with $\theta_g$, $F_1^{\text{cnt}}$ is the combined
counting event-F1 against downward crossings, $R^{\text{det}}$ is the
fraction of ground-truth vehicles matched by a track, $\phi$ is the
fraction of ground-truth observations covered by two or more detections
(fragmentation), and $u$ is the fraction of predicted tracks matching no
ground-truth vehicle. Only grids with a median of at least 25\,FPS in
three serial timing trials and at most 1.5 times the $6{\times}6$ peak
memory were eligible (all were). Among eligible grids within 0.005 of the
best $J$, the one with the fewest cells was chosen. The weights were set
before any cross-validation or test result was computed and were not
changed afterwards; the frozen configuration was recorded before the
test set was assembled.""")

# ---------------- results: remove fig_f1 figure, keep table ----------------
rep_between("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\columnwidth]{figures/fig_f1.pdf}", "Fig.~\\ref{fig:f1} and Table~\\ref{tab:prf} report", "", keep_end=True)
rep("Fig.~\\ref{fig:f1} and Table~\\ref{tab:prf} report the resulting", "Table~\\ref{tab:prf} reports the resulting")

# ---------------- grid subsection ----------------
GRID = open(os.path.join(R, "paper_revision", "scripts", "grid_section.tex"), encoding="utf-8").read()
rep_between("\\subsection{Grid Resolution Sensitivity and Its Impact on Vehicle Classification and Counting}", "\\subsection{Timing: Four Workloads}", GRID, keep_end=True)

# ---------------- benchmark: efficiency, explicit conclusion ----------------
rep("""(Fig.~\\ref{fig:ratios}). The threads-only efficiency
$E(N)=S_{\\text{same}}^{(2)}/N$ is 0.43, 0.22 and 0.11 for $N=2,4,8$. The""", """(Fig.~\\ref{fig:ratios}). At the lane-pool width $N=2$ the threads-only
efficiency $S_{\\text{same}}^{(2)}/N$ is 0.43; other pool widths were not
measured. The""")
rep("""\\subsection{Why Threading Does Not Help Here}
\\label{sec:why}
Three properties""", """\\subsection{Why Threading Does Not Help Here}
\\label{sec:why}
For the tested implementation and workloads, Python threading did not
improve performance. This is a measurement of this code on one machine,
not evidence that Python threading is generally slower. Three properties""")

# ---------------- discussion + threats + conclusion ----------------
DISC = open(os.path.join(R, "paper_revision", "scripts", "discussion.tex"), encoding="utf-8").read()
rep_between("\\subsection{Threats to Validity}", "\\section{Conclusion}", DISC, keep_end=True)
CONC = open(os.path.join(R, "paper_revision", "scripts", "conclusion.tex"), encoding="utf-8").read()
rep_between("\\section{Conclusion}", "\\begin{thebibliography}", CONC, keep_end=True)

# remove the illustrative output overlay (moved to supplementary material)
i = s.index("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\columnwidth]{figures/fig_overlay.jpg}")
j = s.index("\\end{figure}", i) + len("\\end{figure}\n")
s = s[:i] + s[j:]
s = re.sub(r",? and Fig\.~\\ref\{fig:overlay\}\s*the annotated output of the two-lane parallel implementation", "", s)
assert "fig:overlay" not in s, "overlay ref left"
rep("""\\end{tikzpicture}%
}
\\caption{Per-frame dispatch""", """\\end{tikzpicture}%
}%
\\caption{Per-frame dispatch""")
rep("""(Pearson $r=-0.45$, $n=12$) and is lowest in the heavy tier
($1.08\\pm0.20$ vs.\\ $1.34$ light and $1.35$ medium). The cause is open:
the tracking stage takes only 1.6--4.2\\% of sequential run time on every
video, with no density trend, and sequential loop time does not rise
with tier (11.4, 8.4, 8.4\\,ms).""", """(Pearson $r=-0.45$, $n=12$; heavy tier $1.08\\pm0.20$). The cause is
open: tracking takes only 1.6--4.2\\% of sequential run time, with no
density trend.""")
# remaining "test videos" naming for the 12 benchmark videos
rep("matching confusion matrices and F1 scores on all 12 test videos", "matching confusion matrices and F1 scores on all 12 benchmark videos", count=0) if "all 12 test videos" not in s else None
s = s.replace("on the same ROI, their confusion matrices should", "on the same ROI, their confusion matrices should")
open(os.path.join(R, "paper_revision", "revised_paper.tex"), "w", encoding="utf-8").write(s)
print("written", len(s))
