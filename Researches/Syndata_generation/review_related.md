\section{Related Work}

Meta-surveys, also known as tertiary studies or surveys of surveys, systematically 
aggregate data and information from multiple existing secondary studies 
(i.e., systematic literature reviews, surveys, mapping studies, taxonomies) 
over a specific topic~\cite{kotti2023machine}. This methodological approach 
has garnered significant attention across various scientific fields as it 
enables researchers to identify applicable methods and research questions while 
revealing patterns across large numbers of primary studies~\cite{kotti2023machine}. 
Through systematic collection, assessment, analysis, and categorization of existing secondary 
research~\cite{kotti2023machine}, tertiary studies facilitate comprehensive 
synthesis of knowledge that would be difficult to achieve through individual secondary 
studies alone~\cite{lautrup2024systematic,hernandez2022systematic,budu2024evaluation,kaabachi2025scoping}.

\subsection{Tertiary Studies and Meta-Reviews in ML and Data Privacy}

To contextualize our contribution within the contemporary academic landscape, 
we examine existing tertiary studies in machine learning and data privacy, as well 
as surveys pertaining to synthetic data production across several application domains.

\textbf{Tertiary studies in machine learning.} 
Several tertiary studies have systematically examined the broader landscape 
of machine learning research by aggregating and analyzing findings from secondary sources. 
Kotti et al.~\cite{kotti2023machine} conducted a comprehensive tertiary review on the use 
of machine learning in software engineering, synthesizing evidence from 83 secondary studies encompassing 6,117 primary works. 
Their investigation demonstrated the utility of tertiary studies in exposing prevailing research trends, 
assessing the methodological rigor of secondary studies, and identifying underexplored or emerging areas of inquiry. 
Although these efforts establish a solid foundation for systematic tertiary research in machine learning, 
none offers a comprehensive and focused examination of synthetic data generation across multiple domains.

While tertiary studies examining the broader landscape of machine learning research remain limited, 
Kotti et al.~\cite{kotti2023machine} conducted a comprehensive tertiary review on the use of machine learning 
in software engineering, synthesizing evidence from 83 secondary studies encompassing 6,117 primary works. 
Their investigation demonstrated the utility of tertiary studies in exposing prevailing research trends, 
assessing the methodological rigor of secondary studies and identifying underexplored or emerging areas of inquiry. 
This work establishes a methodological foundation for systematic tertiary research; however, to the best of our knowledge, 
no tertiary study has comprehensively examined synthetic data generation across multiple domains.

\textbf{Tertiary studies in data privacy and synthetic data generation.}
The convergence of data privacy and synthetic data generation has garnered
substantial attention in the survey literature, particularly within domain-specific contexts.
Several surveys examine privacy-preserving synthetic data generation techniques,
with emphasis on differential privacy guarantees through frameworks that integrate differential privacy mechanisms
 into generative models, federated learning strategies
and analyses of the privacy–utility trade-off examining how privacy guarantees impact data utility 
across different synthetic data generation approaches.
However, these studies largely focus on privacy mechanisms rather than presenting a 
thorough overview of synthetic data production strategies across domains.
Existing surveys on synthetic data generation can be broadly categorized into two categories: established domains and developing application areas.
In established domains, the medical and healthcare sector has created a significant
corpus of survey research. Prior works investigate synthetic medical imagery created
using GAN-based techniques for generating realistic medical images while preserving patient privacy, privacy-preserving synthetic
electronic health records utilizing methods such as variational autoencoders and generative adversarial networks, and synthetic data for clinical trial simulation to address patient recruitment challenges and improve trial design.
The financial sector demonstrates a similar level of maturity, with surveys on synthetic data
for fraud detection that employ deep learning approaches to generate realistic fraudulent transaction patterns, credit scoring applications using synthetic data to address class imbalance and enhance model robustness, and market simulation through agent-based modeling and synthetic time series generation. Computer vision represents another
saturated area, spanning surveys on synthetic image generation using diffusion models and GANs, video synthesis techniques for action recognition and autonomous driving applications, and data augmentation techniques that leverage synthetic data to improve model generalization.
Collectively, these mature areas—medical and healthcare, financial services, and computer vision—are characterized by:
\begin{enumerate}
\item well-established evaluation criteria and standards;
\item standardized methodological processes;
\item considerable empirical validation;
\item comparably robust theoretical basis.
\end{enumerate}
In emerging domains, survey activity has begun to address synthetic data generation for legal document
synthesis using large language models to generate privacy-compliant legal texts, manufacturing process simulation through digital twin technologies and synthetic sensor data generation, social science research employing synthetic populations for agent-based modeling and survey simulation, educational data generation for learning analytics and adaptive learning systems, and environmental modeling using synthetic climate and ecological data for prediction and scenario analysis.
These fields exhibit:
\begin{enumerate}
\item narrowly scoped issue settings;
\item various and insufficiently standardized analytical methods;
\item changing and frequently incomplete evaluation practices;
\item limited empirical validation on real-world datasets.
\end{enumerate}
\textbf{Methodological approaches in existing surveys.}
Across domains, survey methodologies exhibit substantial variation in rigor and transparency.
Some works adhere to systematic review protocols following PRISMA guidelines for comprehensive literature identification and quality assessment, while others rely on narrative or scoping methodologies that provide broader overviews without systematic selection criteria.
Quality assessment practices are likewise inconsistent: certain surveys adopt structured evaluation criteria
based on established checklists and risk-of-bias tools,
whereas many provide no explicit quality appraisal or rely on informal judgment of study relevance and validity.
This methodological heterogeneity complicates cross-domain comparison and limits the synthesis of higher-level insights.


###########################################################33
\textbf{Tertiary studies in data privacy and synthetic data generation.}
The convergence of data privacy and synthetic data generation has garnered substantial attention in the survey literature, 
particularly within domain-specific contexts. Several surveys examine privacy-preserving synthetic data generation techniques 
focused on differential privacy guarantees, where formal privacy mechanisms add noise to ensure bounded information disclosure^{[s4424802500022w]}, 
federated learning strategies that enable decentralized data generation while maintaining privacy by training across multiple 
clients without sharing raw data^{[ijpds082158]}, and analyses of the privacy–utility trade-off examining how privacy 
protection measures impact data utility^{[s41746024013593,s4424802500022w]}. However, these studies largely focus on privacy 
mechanisms rather than presenting a thorough overview of synthetic data production strategies across domains.
Existing surveys on synthetic data generation can be broadly categorized into two categories: established domains and developing application areas. 
In established domains, the medical and healthcare sector has created a significant 
corpus of survey research^{[1s2.0S0925231222004349main,Synthetic_tudies,s41746024013593]}. 
Prior works investigate synthetic medical imagery created using GAN-based techniques for clinical applications across brain, 
cancer, ophthalmology, and lung studies^{[Synthetic_tudies]}, 
privacy-preserving synthetic electronic health records utilizing methods such as variational autoencoders and generative adversarial 
networks^{[1s2.0S0925231222004349main,s41746024013593]}, and synthetic data 
for clinical trial simulation^{[s41746024013593]}. The financial sector demonstrates a
 similar level of maturity, with surveys on synthetic data for fraud detection, 
 credit scoring, and market simulation^{[3704437]}. 
 Computer vision represents another saturated area, spanning surveys on synthetic 
 image generation using GANs and diffusion models, video synthesis, and data augmentation techniques^{[Synthetic_tudies]}.
Collectively, these mature areas—specifically medical and healthcare, financial services, and computer vision—are characterized by:
\begin{enumerate}
\item well-established evaluation criteria and standards^{[1s2.0S0925231222004349main,Synthetic_tudies]};
\item standardized methodological processes^{[1s2.0S0925231222004349main]};
\item considerable empirical validation^{[s41746024013593]};
\item comparably robust theoretical basis^{[s4424802500022w]}.
\end{enumerate}

In emerging domains, survey activity has begun to address synthetic data generation 
for diverse application areas including manufacturing process simulation through digital 
twin technologies and synthetic sensor data^{[2411.04281v2]}, synthetic populations 
for agent-based modeling in social science research^{[2309.12380v3]}, synthetic patient groups 
and simulation data for educational purposes in healthcare training^{[2411.04281v2]}, and 
applications across environmental modeling and climate simulations^{[3704437]}. 
These fields exhibit:
\begin{enumerate}
\item narrowly scoped issue settings;
\item various and insufficiently standardized analytical methods;
\item changing and frequently incomplete evaluation practices^{[3704437]};
\item limited empirical validation on real-world datasets^{[1s2.0S0925231222004349main]}.
\end{enumerate}
\textbf{Methodological approaches in existing surveys.}
Across domains, survey methodologies exhibit substantial variation in rigor and transparency. 
Some works adhere to systematic review protocols following established PRISMA 
(Preferred Reporting Items for Systematic Reviews and Meta-Analyses) 
guidelines for comprehensive literature identification and selection^{[2309.12380v3,Synthetic_tudies,s41746024013593]}, 
while others rely on narrative or scoping methodologies such as PRISMA-ScR (PRISMA extension for Scoping Reviews) 
that provide broader overviews without systematic selection criteria^{[2411.04281v2,ijpds082158]}. 
Quality assessment practices are likewise inconsistent: certain surveys adopt structured evaluation 
criteria based on frameworks from the Cochrane Handbook for Systematic Reviews of Interventions and 
GRADE (Grading of Recommendations, Assessment, Development and Evaluations) framework^{[2309.12380v3]}, 
assessing risks of selection bias, reporting bias, and performance bias^{[2309.12380v3]}, 
whereas many provide no explicit quality appraisal or rely on informal judgment of study relevance and 
validity^{[3704437]}. This methodological heterogeneity complicates cross-domain comparison and limits 
the synthesis of higher-level insights^{[1s2.0S0925231222004349main,3704437]}.
