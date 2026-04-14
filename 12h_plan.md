Here's a concrete 12-hour plan built around what works for you (short blocks, figure-driven writing, external deadline pressure):

# Evening 1 (~6 hours) — Figures + Results skeleton

- Hours 1-2: Get the quick-win figures done. Combine Fig 2 panels. Aggregate specparam output for Fig 3 (regional bar/dot plots). Generate Fig S2 (R² distributions). These are visualization tasks on existing data — no new computation.

- Hours 3-4: Run the Frauscher peak detection (KS tests) for Fig 4. This is the core analysis you need. Once you have Fig 4 for both iEEG and Mantini, you can compute Fig 5 (cross-modal scatter) and Fig 6 (peak agreement) directly from those outputs.

- Hours 5-6: For each figure you've produced, write one paragraph of results. Use your figure plan — it already has the paragraph template ("Fig 2 → Clustering reproduces Frauscher; HD-EEG shows fewer clusters due to spatial smoothing"). Fill in the actual numbers. Don't polish, just get the result stated with the figure reference.

# Evening 2 (~6 hours) — Aperiodic correction + Discussion + Send

- Hours 7-8: Aperiodic correction analysis (Fig 7). Subtract specparam aperiodic fit, re-run KS detection. Do IRASA correction if time allows, otherwise report specparam only and note IRASA as planned. Write the Fig 7 results paragraph.

- Hour 9: Fig 8 (does correction improve cross-modal agreement). This falls out of Fig 6 + Fig 7 data. Write its paragraph.

- Hours 10-11: Discussion. You already have the section headers and comment-block outlines. Convert each comment block into 1-2 paragraphs of prose. Don't write a perfect discussion — write the argument. You can say "these results suggest X, though this interpretation is qualified by Y" and move on.
- Hour 12: Fill in the remaining TODOs (sensitivity check outcome, electrode count, expand intro comment). Read through once. Fix the conclusion paragraph. Send.

What you explicitly skip for now: Fig 1 (pipeline diagram — cosmetic, do last or in revision), Fig S1 (Nencki/EO-EC — blocked anyway), the Shou dataset entirely, and any attempt to make figures publication-ready. Functional figures with correct data are enough for your supervisor.

One practical note from your therapy session: Set a Focusmate session for each evening block. That's your external accountability. And if you get stuck on a figure for more than 30 minutes, write down what you don't know  and move to the next one. A chapter with 6 figures and written results is infinitely better than a chapter with 9 perfect figures and no prose.
