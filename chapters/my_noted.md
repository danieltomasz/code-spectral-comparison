
- second  block 2A)  - what is the role  of  accounting for  spectral broadband (removing 1/f background of knee like  @afnan2024ValidationMEGSource did) . How to choose between method (fixed/knee) and doed accounting/removing  the 1/f component change/improve the matching? Will looking directly at the proportion of parametrized peak in the band  work better tha unsupervised clustering?

- Second 2B)  We know that Aperiodic features arent comparable between model (knee vs fixed), but do correlate with each  other across modelities? Do aperiodic features extracted from the same dataset AND THE SAME MODEL but with different parameters correlate with each other (for example 1-80 vs 1-45 frequecy range)? we try to quantify IT using the HIERARCHICAL bayesian model   account for Roi and subject  AS RANDOM EFFECTS (WITH THE MAIN EFECT FREQUENCY). How different ARE THE VALUES  EXTRACTED WHEN USING OTHER METHOD (FOR EXAMPLE IRASA)
-

To answer these questions, I compare spectral features across the Frauscher iEEG atlas and source-reconstructed HD-EEG data.

- I examine phenomenologically both oscillatory and broadband components of the power spectrum, using the original  DATA DRIVEN CLASTERING Frauscher-style peak detection procedure for learning about specific frequencies where the power is above estimated 1/f cluster.
 this allow to examine something called natural frequency without using  frquency band, however, it is unstable  (we noticed that the clusteriation itself is unstable filting also outside the frequence range change the no-peak and substantially the shape of obtaine result)

 THe the proceed to TO THE  ANALYSIS OF POWER IN BAND USING REPLICATING AFNAN APPROACH  BY DIVIDING POWER BY POWER IN BANDS (SO FAR WITHOUT REMOVING APERIODIC PART ) AND THEIR RELATIONSHIP USING METRIC defined by afnan paper and will just look for Correlating scalp EEG to intracranial map (figure 3 from @janiukstyte2023NormativeBrainMapping)

 then i will compute 2 model knee and fixed for  1-80 for ieeg and hd eeg data and use bic to choose which is better. i will remove 1/f  or knee from the orginal data and the recompute Afnan metric and their corespondence
This part i will discuss the diference of removing log of PSD - Log (PSD_ap) so in the result is ratio of oscillations vs backround , and the second way would be to compute linear difference od PSD and the aperiodic fit transformed to linear space

 the last part is i will resue my results for fixed (if fixed was choosed) from hd eeg and aslo compute the values for fixed for 1-45 values, then i will compute bayesian hierarchical regression to see where spatially is difference (maybe for knee as well in intracranial )
 THen i add irasa and do similar model  for 1-80 and 1-45 (but i can put them to the same model), but i can phenomenologicaly compare at least so far
