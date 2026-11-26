\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IEEEtran}[2024/01/01 stand-in for layout preview only -- NOT the IEEE class]
\DeclareOption*{}
\ProcessOptions
\LoadClass[10pt,twocolumn,letterpaper]{article}
\RequirePackage{times}
\RequirePackage[letterpaper,top=0.75in,bottom=1in,left=0.625in,right=0.625in,columnsep=0.25in]{geometry}
\RequirePackage[font=footnotesize,labelsep=period,skip=4pt]{caption}
\RequirePackage{titlesec}
\renewcommand\thesection{\Roman{section}}
\renewcommand\thesubsection{\Alph{subsection}}
\titleformat{\section}{\centering\normalsize\scshape}{\thesection.}{0.5em}{}
\titlespacing*{\section}{0pt}{1.4ex plus .4ex}{0.8ex}
\titleformat{\subsection}{\normalsize\itshape}{\thesubsection.}{0.5em}{}
\titlespacing*{\subsection}{0pt}{1.0ex plus .3ex}{0.4ex}
\renewcommand\thetable{\Roman{table}}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\intextsep}{6pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\abovedisplayskip}{4pt}
\newcommand{\IEEEauthorblockN}[1]{#1\par}
\newcommand{\IEEEauthorblockA}[1]{\begin{tabular}[t]{c}\footnotesize #1\end{tabular}}
\newenvironment{IEEEkeywords}{\small\bfseries\itshape Index Terms---\normalfont\bfseries}{\par}
\renewenvironment{abstract}{\small\bfseries\itshape Abstract---\normalfont\bfseries}{\par}
\makeatletter
\renewcommand\maketitle{\twocolumn[{\centering{\fontsize{22}{26}\selectfont\@title\par}\vspace{8pt}{\begin{tabular}[t]{c}\@author\end{tabular}\par}\vspace{10pt}}]}
\makeatother
