%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This iittpdiss.cls is a modified version of iitmdiss.cls where we have included
% more details about how to write your thesis. This class must 
% be included in your thesis to write any thesis format at IIT Tirupati.
% This is a class for LaTeX2e that should format a report as per
% the IIT Tirupati Thesis guidelines.  The page layout is set using the
% geometry package.  The double/singlespacing is setup using
% setspace.  Figure captions are set using the hang option of
% caption2.  natbib is used to do the citation using an
% author-year format. I've made an iittp.bst file that formats 
% the references as per the IIT Tirupati guidelines.
%
% To use this class do the following:
%  \documentclass[options]{iittpdiss} 
%
% Options:
%
%   PhD, MS, MTech, MSc, Dual Degree or BTech -- generates the appropriate title page
%   and loads the report class.
%
%   synopsis -- Generates the title page for the synopsis.  This also
%   loads the article class instead of the report.
%  
% Example:
%
% \documentclass[PhD,synopsis]{iittpdiss}
% \documentclass[MS]{iittpdiss}
%
% IMPORTANT NOTICE:
% 
%   PLEASE DO NOT MESS WITH THE MARGINS AND GENERAL FORMATTING SETUP
%   IN THIS FILE UNLESS YOU ARE ABSOLUTELY SURE THAT THE FORMAT DOES NOT
%   MATCH WITH THE THESIS GUIDELINES.  FOR EXAMPLE, DO NOT CHANGE THE 
%   MARGINS AND SPACING JUST TO MAKE YOUR THESIS LARGER!
% 
% Notes:
%
%   * I am using as much of the Thesis guidelines for the spacing
%     and margins as I can.
%   * I have used iitmdiss.cls from P. Ramachandran
%     as a reference and a source for some of the macros.
%   * This class will assume a4paper with 12pt fonts.
%   * I am no TeXpert so feel free to clean up the mess.
%
% Panchatcharam Mariappan (panch.m@iittp.ac.in)
% March 2019.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{iittpdiss}[2019/03/13 Class for IIT-Tirupati Dissertations]

\newif\if@PhD
\newif\if@DD
\newif\if@MS
\newif\if@MTech
\newif\if@MSc
\newif\if@BTech
\newif\if@Synopsis
\newif\if@secondauthor \@secondauthorfalse
\newif\if@thirdauthor \@thirdauthorfalse

\DeclareOption{PhD}{\@PhDtrue}
\DeclareOption{MS}{\@MStrue}
\DeclareOption{MSc}{\@MSctrue}
\DeclareOption{MTech}{\@MTechtrue}
\DeclareOption{BTech}{\@BTechtrue}
\DeclareOption{DD}{\@DDtrue}
\DeclareOption{Synopsis}{\@Synopsistrue}
\ProcessOptions\relax


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Load the correct class.
\if@Synopsis
    \LoadClass[12pt,a4paper]{article}
\else
    \LoadClass[12pt,a4paper]{report}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Geometry and page layout
\RequirePackage[%compat2,
a4paper,left=1.2in,right=0.8in,top=1in,bottom=0.667in,
                nohead]{geometry}[2002/07/08]
\parskip 8pt

\renewcommand{\@pnumwidth}{2.5em}
%\renewcommand{\@tocrmarg}{2.5em}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Using setspace to do the single/double spacing.  Setspace also
%% makes all tables and figures have single spacing automatically.
\RequirePackage{authblk}
\RequirePackage{setspace} 
\RequirePackage{atbeginend} % used to redefine the list environs
\setstretch{1.5}

\BeforeBegin{itemize}{\vspace*{-2ex}}
\AfterEnd{itemize}{}

\BeforeBegin{enumerate}{\vspace*{-2ex}}
\AfterEnd{enumerate}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The figure captioning is done correctly via the hang option.
%%\RequirePackage[hang]{caption2}  -> TeX generates error
\RequirePackage[hang]{caption}
\RequirePackage{tikz}
\RequirePackage{graphicx}
\RequirePackage[absolute,overlay]{textpos}
\usetikzlibrary{calc}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Bibliography
% 
% natbib allows easy switching of citation style and using it along
% with the iittp.bst makes things perfect.
%\RequirePackage[authoryear,round]{natbib} % for author,yr format
\RequirePackage[numbers,square]{natbib} % for numeric format
\bibliographystyle{iittp} % not working in local
%\bibliographystyle{ieeetr} % if not working use as backup

\newcommand{\bibstyle@iittp}{
  %\bibpunct{(}{)}{;}{a}{,}{,} %author,yr format
  \bibpunct{[}{]}{,}{n}{}{,} %numeric format
\gdef\NAT@biblabel##1{##1. }}
%
\setlength{\bibhang}{0pt}
%\citestyle{iittp}
%\citestyle{unsrt_newen}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Other useful macros.  Many of these are modifications from newdiss.cls

% \department{department name} -- user macro to declare name of department.
\def\department#1{\gdef\@department{\normalsize#1}}

\def\rollnumber#1{\gdef\@rollnumber{#1}}
\def\supervisors#1{\gdef\@supervisors{#1}}
\def\firstauthor#1{\gdef\@firstauthor{#1}}
\def\secondauthor#1{\gdef\@secondauthor{#1}\@secondauthortrue}
\def\thirdauthor#1{\gdef\@thirdauthor{#1}\@thirdauthortrue}
\def\firstrollnumber#1{\gdef\@firstrollnumber{#1}}
\def\secondrollnumber#1{\gdef\@secondrollnumber{#1}}
\def\thirdrollnumber#1{\gdef\@thirdrollnumber{#1}}
%\RequirePackage[dvips]{graphicx}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The title page

\def\maketitle{%
  \pagenumbering{roman}
  \thispagestyle{empty}
  \newpage
  \null
  \setcounter{page}{0}
  \parskip=0pt
  \begin{center}%
  \let \footnote \thanks
  \vglue 0in % this makes top margin 2in
  \vskip -5ex%
  \if@Synopsis
    {\large \em SYNOPSIS OF \par}
  \fi
    {\Large \bf \expandafter{\@title} \par}%
    \vskip 5ex%
    {\normalsize
      \if@PhD
         {\em A THESIS }\par
         \vskip 8ex%
         \if@Synopsis
            {\normalsize \em to be submitted by \par}
         \else
            {\normalsize \em submitted by \par}
         \fi
         \vskip 5ex%
         {\bf \large\MakeUppercase{\@firstauthor}\par\par}
         \vskip 8ex%
         {\em for the award of the degree }\par
         \vskip 8ex%
         {\em of} \par
         \vskip 8ex%
         {\bf \large \MakeUppercase{Doctor of Philosophy}} \par
       \else
         \if@MS
            {\em A THESIS }\par
            \vskip 8ex%
            {\normalsize \em submitted by \par}
            \vskip 8ex%
            {\bf \large\MakeUppercase{\@firstauthor}\par\par}
            \vskip 8ex%
            {\em for the award of the degree }\par
            \vskip 8ex%
            {\em of} \par
            \vskip 8ex%
            {\bf \large \MakeUppercase{Master Of Science}} \par
            (by Research) \par
         \else
%           {\em A Project Report }\par
%           \vskip 5ex%
%           {\normalsize \em submitted by \par}
           \vskip 5ex%
           {\normalsize \em submitted in partial fulfillment of the requirements\par
             for the degree of \par}
           \vskip 5ex%
           \if@MSc
			{\bf MASTER OF SCIENCE}
			\fi           
           \if@MTech
             {\bf MASTER OF TECHNOLOGY}
           \fi
           \if@BTech
              {\bf BACHELOR OF TECHNOLOGY}
           \fi
           \if@DD
			{\bf DUAL DEGREE}
			\fi
           \par
           {\em in }\par
           {\bf  \@department}
           \par
           {\em by }\par
           \vskip 5ex%
           \begin{tabular}{ll}
           {\bf \large\MakeUppercase{\@firstauthor}} &
			{\bf \large\MakeUppercase{\@firstrollnumber}}\\
			\if@secondauthor
	           {\bf \large\MakeUppercase{\@secondauthor}} &
				{\bf \large\MakeUppercase{\@secondrollnumber}}\\			
			\fi
			\if@thirdauthor
			           {\bf \large\MakeUppercase{\@thirdauthor}} &
			{\bf \large\MakeUppercase{\@thirdrollnumber}}\\
			\fi
           \end{tabular}
         \fi
                         \vskip 5ex%
{\bf \large Supervisor(s)\\ {\@supervisors}\par}
     \fi

 
    }
  \end{center}%
\begin{textblock*}{16cm}(3cm,24cm) % {block width} (coords) 
    \begin{minipage}{\textwidth}
\vspace{-1.5in}

%  \begin{tikzpicture}[remember picture,overlay]
  %\node[anchor=south west,inner sep=0pt] at ($(current page.south west)+(4cm,6cm)$) {
  \centerline{\includegraphics[scale=0.25]{Pictures/iittplogo.png}}
  %};

 %  \end{tikzpicture}
%  \vspace*{-0.25in}
%	\begin{textblock*}{16cm}(3cm,24cm) % {block width} (coords) 
\begin{center}
	{\bf \large \normalsize DEPARTMENT OF \@department\par}
{\bf \large INDIAN INSTITUTE OF TECHNOLOGY TIRUPATI}
\vglue 0.50em
{\bf \large \@date}\par
\end{center}

\end{minipage}
  	\end{textblock*}
\parskip 8pt
}
\newcommand{\degree}{Degree}
\newcommand{\research}{research }
\newcommand{\thesis}{thesis}
\if@PhD
	\renewcommand\degree{\textbf{Doctor of Philosophy}}
\fi
\if@MS
	\renewcommand{\degree}{\textbf{Master of Science (by Research)}}
\fi
\if@MSc
\renewcommand{\degree}{\textbf{Master of Science}}
\renewcommand{\research}{project}
\renewcommand{\thesis}{report}
\fi
\if@MTech
\renewcommand{\degree}{\textbf{Master of Technology}}
\renewcommand{\research}{project}
\renewcommand{\thesis}{report}
\fi
\if@BTech
\renewcommand{\degree}{\textbf{Bachelor of Technology}}
\renewcommand{\research}{project}
\renewcommand{\thesis}{report}
\fi
\if@DD
\renewcommand{\degree}{\textbf{DUAL DEGREE}}
\renewcommand{\research}{project}
\renewcommand{\thesis}{report}
\fi



\def\titlepage{\newpage\thispagestyle{empty}\c@page\z@}
\def\endtitlepage{\newpage}

\if@Synopsis
   \renewcommand\refname{REFERENCES}
\else
   % QUOTATIONS
   %
   \def\quotations{\chapter*{\centerline{QUOTATIONS}}
   %\addcontentsline{toc}{chapter}{QUOTATIONS}
   }

   % DEDICATION
   %
   \def\dedication{\chapter*{\centerline{DEDICATION}}
   %\addcontentsline{toc}{chapter}{DEDICATION}
   }

   % CERTIFICATE
   %
   %   - this produces a certificate page as per IIT Tirupati Regulations
   %     comment the following lines out to get it working.
   %
   \def\certificate{%% A preliminary chapter
     \clearpage
     \vspace*{36pt}
     \if@PhD
     \centerline{\Large \bf THESIS CERTIFICATE}
     \else
     \if@MS
		 \centerline{\Large \bf THESIS CERTIFICATE}
		\else     
	     \centerline{\Large \bf BONA FIDE CERTIFICATE}
     \fi
       \fi
     \vskip 25pt
     \thispagestyle{empty}
     \typeout{Certificate}
     \setcounter{page}{0}
   }
   \def\declaration{%% A preliminary chapter
	\clearpage
	\vspace*{36pt}
	\centerline{\Large \bf DECLARATION}
	\vskip 25pt
	\thispagestyle{empty}
	\typeout{Certificate}
	\setcounter{page}{0}
}
   % ACKNOWLEDGEMENTS
   %
   \def\acknowledgements{\chapter*{\centerline{ACKNOWLEDGMENTS}}
     \addcontentsline{toc}{chapter}{ACKNOWLEDGEMENTS}
   }

   % ABSTRACT
   %   - this produces pages suitable for an IIT Tirupati Abstract
   \def\abstract{\chapter*{\centerline{ABSTRACT}}
   \addcontentsline{toc}{chapter}{ABSTRACT}
   }

   % GLOSSARY
   %
   \def\glossary{\chapter*{\centerline{GLOSSARY}}
   \addcontentsline{toc}{chapter}{GLOSSARY}
   }

   % ABBREVIATIONS
   %
   \def\abbreviations{\chapter*{\centerline{ABBREVIATIONS}}
   \addcontentsline{toc}{chapter}{ABBREVIATIONS}
   }

   % Use the nomencl package for the NOTATION section.

   % Makes the chapter heading
   \def\@makechapterhead#1{%
     {\parindent \z@ \raggedright \normalfont
       \ifnum \c@secnumdepth >\m@ne
       \begin{center}
         \Large\bfseries \@chapapp\space \thechapter
       \end{center}
       \par\nobreak
       \vskip 20\p@
       \fi
       \interlinepenalty\@M
       \Large \bf
       \begin{center}
         #1\par\nobreak      
       \end{center}
       \vskip 20\p@
       \thispagestyle{empty}
     }}

   % Makes the chapter heading for \chapter* commands
   \def\@makeschapterhead#1{%
     \vspace*{25\p@}%
     {\parindent \z@ \raggedright
       \normalfont
       \interlinepenalty\@M
       \Large \bfseries #1\par\nobreak
       \vskip 20\p@
     }}


   %LIST OF PAPERS
   \def\listofpapers{
           \normalbaselines
           \chapter*{\centerline{LIST OF PUBLISHED PAPERS BASED ON THESIS}}
           \markboth{LIST OF PAPERS BASED ON THESIS}{LIST OF PAPERS BASED ON THESIS}
           %\addcontentsline{toc}{chapter}{LIST OF PAPERS BASED ON THESIS}
   }

   
   % CV
   \def\CV{\chapter*{\centerline{CURRICULUM VITAE}}}

   % Comittee
   \if@PhD
       \def\committee{\chapter*{\centerline{DOCTORAL COMMITTEE}}}
       \def\conclusion{Concluding Remarks}
	\else 
   \if@MS
      \def\committee{\chapter*{\centerline{MASTER COMMITTEE}}}
      \def\conclusion{Concluding Remarks}
    \else 
       \def\conclusion{Summary}
    \fi
	\fi

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Renewed commands to set the titles of various pages correctly.
   \renewcommand\contentsname{\centering TABLE OF CONTENTS}
   \renewcommand\listfigurename{\centering LIST OF FIGURES}
   \renewcommand\listtablename{\centering LIST OF TABLES}
   \renewcommand{\chaptername}{CHAPTER}
   \renewcommand\bibname{\centering REFERENCES}
   \renewcommand\appendixname{APPENDIX}
\fi


