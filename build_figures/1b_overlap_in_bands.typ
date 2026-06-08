// Simple numbering for non-book documents
#let equation-numbering = "(1)"
#let callout-numbering = "1"
#let subfloat-numbering(n-super, subfloat-idx) = {
  numbering("1a", n-super, subfloat-idx)
}

// Theorem configuration for theorion
// Simple numbering for non-book documents (no heading inheritance)
#let theorem-inherited-levels = 0

// Theorem numbering format (can be overridden by extensions for appendix support)
// This function returns the numbering pattern to use
#let theorem-numbering(loc) = "1.1"

// Default theorem render function
#let theorem-render(prefix: none, title: "", full-title: auto, body) = {
  if full-title != "" and full-title != auto and full-title != none {
    strong[#full-title.]
    h(0.5em)
  }
  body
}
// Some definitions presupposed by pandoc's typst output.
#let content-to-string(content) = {
  if content.has("text") {
    content.text
  } else if content.has("children") {
    content.children.map(content-to-string).join("")
  } else if content.has("body") {
    content-to-string(content.body)
  } else if content == [ ] {
    " "
  }
}

#let horizontalrule = line(start: (25%,0%), end: (75%,0%))

#let endnote(num, contents) = [
  #stack(dir: ltr, spacing: 3pt, super[#num], contents)
]

#show terms.item: it => block(breakable: false)[
  #text(weight: "bold")[#it.term]
  #block(inset: (left: 1.5em, top: -0.4em))[#it.description]
]

// Some quarto-specific definitions.

#show raw.where(block: true): set block(
    fill: luma(230),
    width: 100%,
    inset: 8pt,
    radius: 2pt
  )

#let block_with_new_content(old_block, new_content) = {
  let fields = old_block.fields()
  let _ = fields.remove("body")
  if fields.at("below", default: none) != none {
    // TODO: this is a hack because below is a "synthesized element"
    // according to the experts in the typst discord...
    fields.below = fields.below.abs
  }
  block.with(..fields)(new_content)
}

#let empty(v) = {
  if type(v) == str {
    // two dollar signs here because we're technically inside
    // a Pandoc template :grimace:
    v.matches(regex("^\\s*$")).at(0, default: none) != none
  } else if type(v) == content {
    if v.at("text", default: none) != none {
      return empty(v.text)
    }
    for child in v.at("children", default: ()) {
      if not empty(child) {
        return false
      }
    }
    return true
  }

}

// Subfloats
// This is a technique that we adapted from https://github.com/tingerrr/subpar/
#let quartosubfloatcounter = counter("quartosubfloatcounter")

#let quarto_super(
  kind: str,
  caption: none,
  label: none,
  supplement: str,
  position: none,
  subcapnumbering: "(a)",
  body,
) = {
  context {
    let figcounter = counter(figure.where(kind: kind))
    let n-super = figcounter.get().first() + 1
    set figure.caption(position: position)
    [#figure(
      kind: kind,
      supplement: supplement,
      caption: caption,
      {
        show figure.where(kind: kind): set figure(numbering: _ => {
          let subfloat-idx = quartosubfloatcounter.get().first() + 1
          subfloat-numbering(n-super, subfloat-idx)
        })
        show figure.where(kind: kind): set figure.caption(position: position)

        show figure: it => {
          let num = numbering(subcapnumbering, n-super, quartosubfloatcounter.get().first() + 1)
          show figure.caption: it => block({
            num.slice(2) // I don't understand why the numbering contains output that it really shouldn't, but this fixes it shrug?
            [ ]
            it.body
          })

          quartosubfloatcounter.step()
          it
          counter(figure.where(kind: it.kind)).update(n => n - 1)
        }

        quartosubfloatcounter.update(0)
        body
      }
    )#label]
  }
}

// callout rendering
// this is a figure show rule because callouts are crossreferenceable
#show figure: it => {
  if type(it.kind) != str {
    return it
  }
  let kind_match = it.kind.matches(regex("^quarto-callout-(.*)")).at(0, default: none)
  if kind_match == none {
    return it
  }
  let kind = kind_match.captures.at(0, default: "other")
  kind = upper(kind.first()) + kind.slice(1)
  // now we pull apart the callout and reassemble it with the crossref name and counter

  // when we cleanup pandoc's emitted code to avoid spaces this will have to change
  let old_callout = it.body.children.at(1).body.children.at(1)
  let old_title_block = old_callout.body.children.at(0)
  let children = old_title_block.body.body.children
  let old_title = if children.len() == 1 {
    children.at(0)  // no icon: title at index 0
  } else {
    children.at(1)  // with icon: title at index 1
  }

  // TODO use custom separator if available
  // Use the figure's counter display which handles chapter-based numbering
  // (when numbering is a function that includes the heading counter)
  let callout_num = it.counter.display(it.numbering)
  let new_title = if empty(old_title) {
    [#kind #callout_num]
  } else {
    [#kind #callout_num: #old_title]
  }

  let new_title_block = block_with_new_content(
    old_title_block,
    block_with_new_content(
      old_title_block.body,
      if children.len() == 1 {
        new_title  // no icon: just the title
      } else {
        children.at(0) + new_title  // with icon: preserve icon block + new title
      }))

  align(left, block_with_new_content(old_callout,
    block(below: 0pt, new_title_block) +
    old_callout.body.children.at(1)))
}

// 2023-10-09: #fa-icon("fa-info") is not working, so we'll eval "#fa-info()" instead
#let callout(body: [], title: "Callout", background_color: rgb("#dddddd"), icon: none, icon_color: black, body_background_color: white) = {
  block(
    breakable: false, 
    fill: background_color, 
    stroke: (paint: icon_color, thickness: 0.5pt, cap: "round"), 
    width: 100%, 
    radius: 2pt,
    block(
      inset: 1pt,
      width: 100%, 
      below: 0pt, 
      block(
        fill: background_color,
        width: 100%,
        inset: 8pt)[#if icon != none [#text(icon_color, weight: 900)[#icon] ]#title]) +
      if(body != []){
        block(
          inset: 1pt, 
          width: 100%, 
          block(fill: body_background_color, width: 100%, inset: 8pt, body))
      }
    )
}


// syntax highlighting functions from skylighting:
/* Function definitions for syntax highlighting generated by skylighting: */
#let EndLine() = raw("\n")
#let Skylighting(fill: none, number: false, start: 1, sourcelines) = {
   let blocks = []
   let lnum = start - 1
   let bgcolor = rgb("#f1f3f5")
   for ln in sourcelines {
     if number {
       lnum = lnum + 1
       blocks = blocks + box(width: if start + sourcelines.len() > 999 { 30pt } else { 24pt }, text(fill: rgb("#aaaaaa"), [ #lnum ]))
     }
     blocks = blocks + ln + EndLine()
   }
   block(fill: bgcolor, width: 100%, inset: 8pt, radius: 2pt, blocks)
}
#let AlertTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let AnnotationTok(s) = text(fill: rgb("#5e5e5e"),raw(s))
#let AttributeTok(s) = text(fill: rgb("#657422"),raw(s))
#let BaseNTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let BuiltInTok(s) = text(fill: rgb("#003b4f"),raw(s))
#let CharTok(s) = text(fill: rgb("#20794d"),raw(s))
#let CommentTok(s) = text(fill: rgb("#5e5e5e"),raw(s))
#let CommentVarTok(s) = text(style: "italic",fill: rgb("#5e5e5e"),raw(s))
#let ConstantTok(s) = text(fill: rgb("#8f5902"),raw(s))
#let ControlFlowTok(s) = text(weight: "bold",fill: rgb("#003b4f"),raw(s))
#let DataTypeTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let DecValTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let DocumentationTok(s) = text(style: "italic",fill: rgb("#5e5e5e"),raw(s))
#let ErrorTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let ExtensionTok(s) = text(fill: rgb("#003b4f"),raw(s))
#let FloatTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let FunctionTok(s) = text(fill: rgb("#4758ab"),raw(s))
#let ImportTok(s) = text(fill: rgb("#00769e"),raw(s))
#let InformationTok(s) = text(fill: rgb("#5e5e5e"),raw(s))
#let KeywordTok(s) = text(weight: "bold",fill: rgb("#003b4f"),raw(s))
#let NormalTok(s) = text(fill: rgb("#003b4f"),raw(s))
#let OperatorTok(s) = text(fill: rgb("#5e5e5e"),raw(s))
#let OtherTok(s) = text(fill: rgb("#003b4f"),raw(s))
#let PreprocessorTok(s) = text(fill: rgb("#ad0000"),raw(s))
#let RegionMarkerTok(s) = text(fill: rgb("#003b4f"),raw(s))
#let SpecialCharTok(s) = text(fill: rgb("#5e5e5e"),raw(s))
#let SpecialStringTok(s) = text(fill: rgb("#20794d"),raw(s))
#let StringTok(s) = text(fill: rgb("#20794d"),raw(s))
#let VariableTok(s) = text(fill: rgb("#111111"),raw(s))
#let VerbatimStringTok(s) = text(fill: rgb("#20794d"),raw(s))
#let WarningTok(s) = text(style: "italic",fill: rgb("#5e5e5e"),raw(s))



#let article(
  title: none,
  subtitle: none,
  authors: none,
  keywords: (),
  date: none,
  abstract-title: none,
  abstract: none,
  thanks: none,
  cols: 1,
  lang: "en",
  region: "US",
  font: none,
  fontsize: 11pt,
  title-size: 1.5em,
  subtitle-size: 1.25em,
  heading-family: none,
  heading-weight: "bold",
  heading-style: "normal",
  heading-color: black,
  heading-line-height: 0.65em,
  mathfont: none,
  codefont: none,
  linestretch: 1,
  sectionnumbering: none,
  linkcolor: none,
  citecolor: none,
  filecolor: none,
  toc: false,
  toc_title: none,
  toc_depth: none,
  toc_indent: 1.5em,
  doc,
) = {
  // Set document metadata for PDF accessibility
  set document(title: title, keywords: keywords)
  set document(
    author: authors.map(author => content-to-string(author.name)).join(", ", last: " & "),
  ) if authors != none and authors != ()
  set par(
    justify: true,
    leading: linestretch * 0.65em
  )
  set text(lang: lang,
           region: region,
           size: fontsize)
  set text(font: font) if font != none
  show math.equation: set text(font: mathfont) if mathfont != none
  show raw: set text(font: codefont) if codefont != none

  set heading(numbering: sectionnumbering)

  show link: set text(fill: rgb(content-to-string(linkcolor))) if linkcolor != none
  show ref: set text(fill: rgb(content-to-string(citecolor))) if citecolor != none
  show link: this => {
    if filecolor != none and type(this.dest) == label {
      text(this, fill: rgb(content-to-string(filecolor)))
    } else {
      text(this)
    }
   }

  let has-title-block = title != none or (authors != none and authors != ()) or date != none or abstract != none
  if has-title-block {
    place(
      top,
      float: true,
      scope: "parent",
      clearance: 4mm,
      block(below: 1em, width: 100%)[

        #if title != none {
          align(center, block(inset: 2em)[
            #set par(leading: heading-line-height) if heading-line-height != none
            #set text(font: heading-family) if heading-family != none
            #set text(weight: heading-weight)
            #set text(style: heading-style) if heading-style != "normal"
            #set text(fill: heading-color) if heading-color != black

            #text(size: title-size)[#title #if thanks != none {
              footnote(thanks, numbering: "*")
              counter(footnote).update(n => n - 1)
            }]
            #(if subtitle != none {
              parbreak()
              text(size: subtitle-size)[#subtitle]
            })
          ])
        }

        #if authors != none and authors != () {
          let count = authors.len()
          let ncols = calc.min(count, 3)
          grid(
            columns: (1fr,) * ncols,
            row-gutter: 1.5em,
            ..authors.map(author =>
                align(center)[
                  #author.name \
                  #author.affiliation \
                  #author.email
                ]
            )
          )
        }

        #if date != none {
          align(center)[#block(inset: 1em)[
            #date
          ]]
        }

        #if abstract != none {
          block(inset: 2em)[
          #text(weight: "semibold")[#abstract-title] #h(1em) #abstract
          ]
        }
      ]
    )
  }

  if toc {
    let title = if toc_title == none {
      auto
    } else {
      toc_title
    }
    block(above: 0em, below: 2em)[
    #outline(
      title: toc_title,
      depth: toc_depth,
      indent: toc_indent
    );
    ]
  }

  doc
}

#set table(
  inset: 6pt,
  stroke: none
)
#show link: set text(fill: rgb("0000FF"))
#show ref: set text(fill: rgb("0000FF"))
#let brand-color = (:)
#let brand-color-background = (:)
#let brand-logo = (:)

#set page(
  paper: "us-letter",
  margin: (x: 1.25in, y: 1.25in),
  numbering: "1",
  columns: 1,
)

#show: doc => article(
  title: [Overlap in Bands],
  authors: (
    ( name: [Daniel Borek],
      affiliation: [],
      email: [] ),
    ),
  fontsize: 12pt,
  sectionnumbering: "1.1.1",
  toc: true,
  toc_title: [Table of contents],
  toc_depth: 3,
  doc,
)

= Methods
<methods>
=== Canonical-band and normalised relative-power maps
<canonical-band-and-normalised-relative-power-maps>
As a second, complementary approach, we mapped each region's normalised PSD onto the canonical bands defined above and took relative band power per region.

Relative band power does not separate the rhythmic peak from the aperiodic background, so it carries both: a region can show high relative band power because it has a strong rhythm or because its background is steep in that band Castanheira et al. (#link(<ref-castanheira2025QuantifyingRhythmicArrhythmic>)[2025]). Relative band powers are also compositional --- within each region the five bands are normalised to a fixed total, so they are not free to vary independently \(#link(<ref-filzmoser2012CorrelationAnalysisCompositional>)[Filzmoser and Hron 2012]). We therefore report the cross-modal correlations of these maps as descriptive measures of spatial agreement rather than as formal inference, and carry quantitative cross-modal inference with the aperiodic-separated peak descriptors, which are not subject to this closure.

== Quantitative descriptions
<quantitative-descriptions>
=== Cross-modal correspondence on relative PSD spectra
<cross-modal-correspondence-on-relative-psd-spectra>
This is the first of three cross-modal correspondence computations; the subtraction-based and modelled-peak versions are defined below and compared in #strong[?\@sec-concordance]

We quantified cross-modal agreement on the unit-power PSDs using the band-overlap metric of Afnan et al. (#link(<ref-afnan2023ValidatingMEGSource>)[2023]). Channels were pooled across hemispheres by grouping on the hemisphere-agnostic MICCAI region name, collapsing left and right ROIs into the 38 bilateral regions of the atlas. Within each region $r$, we treated the iEEG atlas as the reference modality and the HD-EEG source data as the estimated modality. Let $tilde(L)_r \( f \)$ denote the median log-PSD across the channels of region $r$ at frequency bin $f$, and $upright("SD")_r^(thin upright("HD")) \( f \)$ the corresponding HD-EEG standard deviation. The bin-wise overlap is

$ upright("overlap")_r \( f \) = cases(delim: "{", 1 - frac(#scale(x: 120%, y: 120%)[\|] tilde(L)_r^(thin upright("iEEG")) \( f \) - tilde(L)_r^(thin upright("HD")) \( f \) #scale(x: 120%, y: 120%)[\|], upright("SD")_r^(thin upright("HD")) \( f \)) \, & #scale(x: 120%, y: 120%)[\|] tilde(L)_r^(thin upright("iEEG")) \( f \) - tilde(L)_r^(thin upright("HD")) \( f \) #scale(x: 120%, y: 120%)[\|] lt.eq upright("SD")_r^(thin upright("HD")) \( f \), 0 \, & upright("otherwise.")) $

The metric equals 1 when the iEEG median lands exactly on the HD-EEG median and falls linearly to 0 once the two medians differ by one HD-EEG standard deviation. We averaged the bin-wise overlap across the frequency bins of each canonical band, giving one value per region per band, and report both the bin-level and the band-averaged overlap. To complement the overlap, we also rank-correlated the regional band estimates between modalities across regions (Spearman).

We also recomputed the overlap on the 22 data-driven Frauscher intervals \(#link(<ref-frauscher2018AtlasNormalIntracranial>)[Frauscher et al. 2018]), which give a more fine-grained view than the five canonical bands.

== Relative power after specparam aperiodic part removal
<relative-power-after-specparam-aperiodic-part-removal>
We asked whether separating periodic from aperiodic activity changes the above picture. We addressed it in two linked ways: by modelling oscillatory peaks directly with specparam, and by removing the aperiodic background and recomputing the cross-modal overlap.

Each channel's PSD was parameterised with specparam, the spectral parameterisation model of Donoghue et al. (#link(<ref-donoghue2020ParameterizingNeuralPower>)[2020]), fitted over 1-80 Hz with a peak configuration shared across both modalities: peak width limits of 1-8 Hz, at most 8 peaks, a minimum peak height of 0.1, and a peak threshold of 3 SD. Only the aperiodic component differed between modalities: it was modelled in the form chosen by the per-modality knee-versus-fixed BIC comparison (#strong[?\@sec-aperiodic-selection]) --- a knee model for the iEEG atlas and a fixed (knee-free) model for the source-reconstructed HD-EEG. From each fit we reconstructed the aperiodic component on the fitted frequency grid and subtracted it in linear power (next subsection), then normalised the residual to total power, so each band's oscillatory power is expressed as a fraction of total power and is directly comparable to the uncorrected relative band power above. One fit was produced per channel; goodness of fit and the fitted aperiodic parameters are reported in the Results.

We reconstructed the fitted aperiodic component on the fitted frequency grid and subtracted it from the empirical PSD in linear power space, giving the residual

$ upright(P S D)_(upright(o s c)) \( f \) = upright(P S D) \( f \) - upright(P S D)_(upright(a p)) \( f \) . $

This is the primary residual used throughout. We report a log-ratio alternative, $L_(upright(o s c)) \( f \) = log_10 \( upright(P S D) \( f \) \/ upright(P S D)_(upright(a p)) \( f \) \)$, which detrends in log space, the variant used for detrending in many studies, as a sensitivity comparison.

Both residuals depend on the aperiodic fit, so error in the offset or exponent propagates into the oscillatory estimate; Castanheira et al. (#link(<ref-castanheira2025QuantifyingRhythmicArrhythmic>)[2025]) show this produces biased oscillatory power, more so in log space. Because we select the aperiodic mode per modality (knee vs fixed, #strong[?\@sec-aperiodic-selection]), the iEEG and HD-EEG residuals are produced by different aperiodic models.

=== Assessing the overlap: reliability and rhythm presence
<assessing-the-overlap-reliability-and-rhythm-presence>
The band-overlap of Afnan et al. (#link(<ref-afnan2023ValidatingMEGSource>)[2023]) is a #emph[descriptive] agreement metric: on its own it indicates neither whether the agreement is statistically reliable nor whether there is a rhythm present to agree about --- two flat residuals coincide and score an overlap near 1, a spurious agreement. We therefore assess the overlap in two complementary ways. Both are subject-clustered bootstraps with whole subjects resampled with replacement (iEEG #NormalTok("patient");, HD #NormalTok("dataset");\; B = 2000), but they target different quantities and answer different questions.

We deliberately use a subject-clustered bootstrap rather than the channel-wise Welch #emph[t]-test of Afnan et al. (#link(<ref-afnan2023ValidatingMEGSource>)[2023]), for three reasons specific to our design. First, our channels are nested in a small number of subjects (the source-reconstructed HD-EEG comprises only 19 subjects, each contributing many channels), and source leakage makes neighbouring source-space channels strongly correlated; a #emph[t]-test on channels treats them as independent and so inflates the effective sample size from \~19 to several hundred, anti-conservatively shrinking the standard error. Welch's correction addresses unequal #emph[variance], not this non-independence, so it does not remedy the problem --- whereas resampling whole subjects respects the nesting and keeps the exchangeable unit (the subject) as the unit of inference. Second, relative band power is bounded and the aperiodic-removed residual has a point mass at zero, violating the normality the #emph[t]-test leans on at this sample size; the bootstrap is distribution-free. Third, and most directly, the overlap is a single statistic #emph[derived] from both modalities, not a contrast between two samples, so a two-sample #emph[t]-test does not apply to it at all --- the only way to attach uncertainty to the overlap is to resample and recompute it. The reference (iEEG) modality is resampled too, since the atlas medians are themselves estimated from a finite set of patients.

The first assesses the #strong[reliability of the overlap value]. We resample subjects in both modalities, recompute the band-overlap on each draw, and take a 95% percentile confidence interval per region and band; cells whose interval includes 0 are masked (shown in grey; #ref(<fig-overlap-pair-corrected>, supplement: [Figure])). This asks #emph[is the measured agreement reliably above zero?] Because the metric is bounded in $\[ 0 \, 1 \]$ and has no chance or permutation baseline, it is a reliability check --- that the value is robust rather than zero by sampling noise --- not a test against a null of no correspondence. It is also band-width dependent: averaging a non-negative metric over a wide canonical band drives its lower confidence bound above zero almost everywhere, so the five canonical bands leave far fewer cells masked than the 22 narrow Frauscher intervals, and the two band schemes are not directly comparable on this criterion.

The second assesses whether #strong[a rhythm is present to be compared], using the iEEG atlas as the reference (the ground truth). We bootstrap the aperiodic-removed band power per region and band and consider a region to carry a rhythm in a band when its mean residual power reliably exceeds zero (95% CI \> 0). The overlap is then shown only where #emph[both] modalities carry a rhythm; cells are masked where neither does (the overlap is uninterpretable --- there is no oscillation to recover), and marked with a cross where the iEEG atlas carries a rhythm that HD-EEG fails to recover (#ref(<fig-overlap-corr-presence-gated-pair>, supplement: [Figure])). This addresses the validation question directly: #emph[where does source-reconstructed HD-EEG recover a genuine iEEG rhythm, and where does it miss one?] Presence here is significant oscillatory power above the $1 \/ f$ background, a more permissive criterion than a fitted specparam peak.

The two analyses are complementary --- the first measures how trustworthy the agreement value is, the second whether the agreement concerns real oscillatory structure --- so we read the presence-conditioned map as the primary validation summary, because it separates genuine rhythm recovery from the spurious agreement of two flat residuals, and the confidence-interval map as a check on the metric itself. Both use the same display, differing only in the masking rule, and we show each band scheme (canonical bands, Frauscher intervals) side by side (#ref(<fig-overlap-pair-corrected>, supplement: [Figure]), #ref(<fig-overlap-corr-presence-gated-pair>, supplement: [Figure])).

Model fits were evaluated using R² and mean absolute error; fits with R² below 0.90 were excluded.

= Results
<results>
#figure([
#box(image("figures/chapter2/relative_power/relpower_heatmaps_uncorrected.pdf"))
], caption: figure.caption(
position: bottom, 
[
Median relative band power per bilateral region, before aperiodic correction, as region × band heatmaps. Modalities are stacked --- iEEG (top, A-B) over HD-EEG (bottom, C-D) --- with the five canonical bands (left) and the 22 Frauscher intervals (right, Frauscher et al. (#link(<ref-frauscher2018AtlasNormalIntracranial>)[2018])) across the columns, so each band scheme aligns vertically between modalities. The value in each cell is the median across the region's channels of the within-band relative power (band power as a fraction of total power, so in \[0, 1\]); each panel has its own colorbar, auto-scaled to its maximum, because the wide canonical bands carry a much larger fraction of total power than the narrow Frauscher bins. Rows are bilateral MICCAI regions grouped by lobe, with lobe-coloured labels.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-relpower-heatmaps>


#figure([
#box(image("figures/chapter2/relative_power/relpower_maps_uncorrected.pdf"))
], caption: figure.caption(
position: bottom, 
[
Group average relative PSD per ROI, faceted by canonical band (columns) and modality (rows: iEEG top, HD-EEG source bottom). Colour is each band's relative power min-max normalised across both modalities (the per-band shared scale of Afnan et al. (#link(<ref-afnan2023ValidatingMEGSource>)[2023]), shown as one 0-1 bar), so the within-band spatial pattern is comparable across modalities; absolute magnitude differs across bands and is not encoded. Hippocampus and amygdala have no cortical-surface geometry in the Frauscher ggsegpy atlas and are not shown.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-relpower-maps-uncorrected>


=== Band-power map and cross-modal overlap
<band-power-map-and-cross-modal-overlap>
We present regional relative band power, compared across modalities with the band-overlap metric.

Regional relative-power maps agreed across modalities most strongly in the alpha band (r = 0.60) and, more weakly, in the delta band (r = 0.42). Theta showed essentially no cross-modal correspondence (r = 0.13), beta only weak agreement (r = 0.30), and gamma was anti-correlated (r = −0.29). The spatial organisation of relative power is therefore only partially preserved between the iEEG atlas and source-reconstructed HD-EEG, and the preservation is band-dependent: it is confined largely to the alpha and low-frequency ranges that carry the clearest oscillatory structure, and breaks down at high frequencies. The negative gamma correlation is consistent with the high-frequency HD-EEG signal being dominated by a shallow aperiodic tail and noise floor rather than by physiological gamma --- the same mechanism that drove the spurious gamma over-segmentation in the interval analysis --- so that regions ranked high in HD-EEG gamma relative power are not those ranked high in the atlas. Relative to the identity line

#figure([
#box(image("figures/chapter2/relative_power/relpower_scatter_uncorrected.pdf"))
], caption: figure.caption(
position: bottom, 
[
Cross-modal correspondence of uncorrected relative band power. Each panel is one canonical band; each point is one bilateral MICCAI region, with source HD-EEG relative power on the x-axis and the iEEG atlas on the y-axis. Points are coloured by lobe, the solid line is the per-band OLS fit, and the dashed line marks identity (y = x). Strip labels give the per-band Pearson
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-relpower-scatter-uncorrected>


These per-band correlations should be read as descriptive rather than inferential. Relative band powers are #emph[compositional]: within each channel they are normalised to a fixed total, so the five bands are not independent and an increase in one band necessarily depresses the others. This closure constraint induces negative coupling between bands by construction --- part of the negative gamma correlation is therefore arithmetic rather than physiological --- and it violates the independence assumptions of ordinary correlation and t-tests applied to the raw proportions\(#link(<ref-filzmoser2012CorrelationAnalysisCompositional>)[Filzmoser and Hron 2012]). Valid inference would require analysing the maps in a log-ratio space (e.g.~centred or isometric log-ratio transforms), which also removes any dependence on the per-channel normalisation choice. We retain the untransformed maps here as a descriptive summary and defer formal cross-modal inference to the aperiodic-separated peak descriptors below, which are not subject to the same closure.

In #ref(<fig-relpower-maps-uncorrected>, supplement: [Figure]) we present results on scalp.

#figure([
#box(image("figures/chapter2/relative_power/overlap_maps_uncorrected.pdf"))
], caption: figure.caption(
position: bottom, 
[
Cross-modal spectral overlap between the iEEG atlas and source-reconstructed HD-EEG mapped onto the cortex, one panel per canonical band (left hemisphere, lateral and medial views), computed on uncorrected relative-power PSDs. Each bilateral MICCAI region is coloured by its band-averaged Afnan overlap on a fixed 0-1 scale --- the metric is already bounded, so colour is comparable across bands and across regions; the strip reports the per-band mean overlap across regions. Hippocampus and amygdala have no cortical-surface geometry in the Frauscher ggsegpy atlas and are not shown.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-overlap-maps-uncorrected>


We recreated simlat plot as in the clustering part to be able to compare. (we retrive a strong delta and alpha with some regions with string beta ). We also tested again presence of no-peaks (which is imprtant later) but we also butstraped aggainst 0 difference

#figure([
#box(image("figures/chapter2/relative_power/overlap_pair_uncorrected.pdf"))
], caption: figure.caption(
position: bottom, 
[
Uncorrected cross-modal overlap. A) five canonical EEG bands B)the 22 intervals defined in Frauscher et al. (#link(<ref-frauscher2018AtlasNormalIntracranial>)[2018]). Both panels are masked (grey) where the bootstrap 95% CI of the overlap includes 0 and carry a contrast dot where a reliable overlap sits on a flat residual (no significant rhythm in either modality --- spurious agreement).
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-overlap-pair-uncorrected>


=== Aperiodic-removed band power
<aperiodic-removed-band-power>
The specparam model captured the aperiodic background well in the iEEG atlas: across the 1772 channels the knee model reached a mean R² of 0.98 (± 0.01; median 0.985, only 4 channels below 0.90), with a mean offset of 5.10 ± 1.57 and an exponent of 3.39 ± 0.88, and a median knee frequency of 11.2 Hz (IQR 7.8-15.9). The fixed model on the 1444 source-reconstructed HD-EEG channels fit less well and far more variably (mean R² 0.91 ± 0.11, median 0.94; 376/1444 channels, 26%, fell below 0.90), with a much lower offset (−2.79 ± 0.58) and a shallower exponent (0.94 ± 0.33). Mean absolute error was comparable across modalities (0.087 vs 0.073 in log₁₀ power). The substantially poorer and more dispersed HD-EEG fits, together with the shallow exponent, indicate that the HD-EEG aperiodic estimate --- and hence its oscillatory residual --- is less reliable than the iEEG one.

The same three views, repeated on the #emph[periodic] part of the spectrum. Each channel is fit with #NormalTok("specparam"); in its BIC-selected aperiodic mode (iEEG knee, HD-EEG fixed;), the fitted aperiodic component on narmalized spectrum is subtracted, so the per-band value is the channel's oscillatory power in that band as a fraction of total power --- directly comparable to the uncorrected relative band power above. One fit per modality feeds all three figures; the fitted residuals are cached so the fit runs once.

Removing the aperiodic background isolates the rhythmic peaks from the $1 \/ f$ tail, so the correspondence is no longer inflated (or, in gamma, depressed) by the steepness of the background. The closure constraint still applies --- the linear residual is normalised to total power --- so these per-band correlations remain descriptive.

The fitted aperiodic parameters summarise each modality's $1 \/ f$ background --- iEEG fit with a knee, HD-EEG with the fixed (knee-free) model. Offset and exponent are mean ± sd across channels; the knee is reported as the knee #emph[frequency] ($upright("knee")^(1 \/ chi)$, in Hz) as median \[IQR\], since the raw knee parameter is heavy-tailed (a few channels push the knee out of band).

#figure([
#table(
  columns: (16.84%, 14.74%, 16.84%, 15.79%, 22.11%, 13.68%),
  align: (left,right,left,left,left,right,),
  table.header([modality], [n channels], [offset], [exponent], [knee freq (Hz)], [R² (mean)],),
  table.hline(),
  [iEEG (knee)], [1772], [5.096 ± 1.570], [3.392 ± 0.884], [11.23 \[7.76, 15.92\]], [0.982],
  [HD-EEG (fixed)], [1444], [-2.787 ± 0.577], [0.944 ± 0.325], [---], [0.908],
)
], caption: figure.caption(
separator: "", 
position: top, 
[
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-aperiodic-params>


#figure([
#box(image("figures/chapter2/corrected/relpower_scatter_corrected.pdf"))
], caption: figure.caption(
position: bottom, 
[
Cross-modal correspondence of the aperiodic-removed (oscillatory residual) band power. Each panel is one canonical band; each point is one bilateral MICCAI region, source HD-EEG oscillatory residual on x and the iEEG atlas on y. Points are coloured by lobe, the solid line is the per-band OLS fit, and the dashed line marks identity (y = x). The residual is the specparam linear oscillatory power (band fraction of total) with the BIC-selected aperiodic component removed.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-relpower-scatter-corrected>


#figure([
#box(image("figures/chapter2/relative_power/relpower_maps_corrected.pdf"))
], caption: figure.caption(
position: bottom, 
[
Group-average oscillatory residual per ROI (aperiodic component removed), faceted by canonical band (columns) and modality (rows: iEEG top, HD-EEG source bottom). Colour is each band's residual min-max normalised across both modalities (per-band shared scale), so the within-band spatial pattern is comparable across modalities. Hippocampus and amygdala have no cortical-surface geometry in the Frauscher ggsegpy atlas and are not shown.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-relpower-maps-corrected>


#figure([
#box(image("figures/chapter2/relative_power/relpower_heatmaps_corrected.pdf"))
], caption: figure.caption(
position: bottom, 
[
Median oscillatory band power per bilateral region after specparam aperiodic correction --- the aperiodic-removed twin of #ref(<fig-relpower-heatmaps>, supplement: [Figure]), same stacked layout: iEEG (top, A-B) over HD-EEG (bottom, C-D), canonical bands (left) and the 22 Frauscher intervals (right). Each cell is the median across the region's channels of the within-band oscillatory residual (a fraction of total power, so directly comparable to the uncorrected relative band power); each panel has its own colorbar auto-scaled to its maximum. The sub-1 Hz Frauscher interval is blank (the residual is defined on the 1-80 Hz fit grid). Rows are bilateral MICCAI regions grouped by lobe, with lobe-coloured labels.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-relpower-heatmaps-corrected>


#figure([
#box(image("figures/chapter2/relative_power/overlap_pair_corrected.pdf"))
], caption: figure.caption(
position: bottom, 
[
Aperiodic-removed cross-modal overlap. A) five canonical EEG bands B)the 22 intervals defined in Frauscher et al. (#link(<ref-frauscher2018AtlasNormalIntracranial>)[2018]). Both panels are masked (grey) where the bootstrap 95% CI of the overlap includes 0 and carry a contrast dot where a reliable overlap sits on a flat residual (no significant rhythm in either modality --- spurious agreement).
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-overlap-pair-corrected>


=== Presence-conditioned overlap (peaks-presence bootstrap)
<presence-conditioned-overlap-peaks-presence-bootstrap>
As an alternative to conditioning on the overlap's own confidence interval, we condition on whether a rhythm is actually present. This is the same subject-clustered bootstrap of the oscillatory-residual band power used above (#NormalTok("band_presence");, 95% CI \> 0), but here it drives a view of the canonical-band overlap referenced to the iEEG atlas: only the mask-builder changes (#NormalTok("presence_gated_masks"); instead of #NormalTok("overlap_frauscher_masks");), the overlap, presence and renderer are reused.

#figure([
#box(image("figures/chapter2/relative_power/overlap_pair_corrected_presencegated.pdf"))
], caption: figure.caption(
position: bottom, 
[
Presence-conditioned cross-modal overlap of the aperiodic-removed residual, the two band schemes side by side (composed with #NormalTok("compose_panels");): five canonical EEG bands (left) and the 22 Frauscher intervals (right), at their natural asymmetric width. In both panels the Afnan overlap is coloured only where a subject-clustered bootstrap (whole subjects resampled --- iEEG patients, HD datasets; 95% CI of the mean residual band power excludes 0) finds a rhythm in #strong[both] modalities; cells are masked where no rhythm is present in both and marked × where the iEEG atlas carries a rhythm that HD-EEG did not recover. Same renderer as the confidence-interval pair (#ref(<fig-overlap-pair-corrected>, supplement: [Figure])); only the masking rule differs (presence rather than the overlap's confidence interval), and the band scheme differs between panels.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-overlap-corr-presence-gated-pair>


References

#block[
#block[
Afnan, Jawata, Nicolás von Ellenrieder, Jean-Marc Lina, et al. 2023. “Validating MEG Source Imaging of Resting State Oscillatory Patterns with an Intracranial EEG Atlas.” #emph[NeuroImage] 274 (July): 120158. #link("https://doi.org/10.1016/j.neuroimage.2023.120158").

] <ref-afnan2023ValidatingMEGSource>
#block[
Castanheira, Jason da Silva, Mathieu Landry, and Stephen M. Fleming. 2025. #emph[Quantifying Rhythmic and Arrhythmic Components of Brain Activity]. #link("https://doi.org/10.1101/2025.09.24.678322").

] <ref-castanheira2025QuantifyingRhythmicArrhythmic>
#block[
Donoghue, Thomas, Matar Haller, Erik J. Peterson, et al. 2020. “Parameterizing Neural Power Spectra into Periodic and Aperiodic Components.” #emph[Nature Neuroscience] 23 (12): 1655--65. #link("https://doi.org/10.1038/s41593-020-00744-x").

] <ref-donoghue2020ParameterizingNeuralPower>
#block[
Filzmoser, Peter, and Karel Hron. 2012. “Correlation Analysis for Compositional Data.” #emph[Mathematical Geosciences] 41 (8): 905--19. #link("https://doi.org/10.1007/s11004-008-9196-y").

] <ref-filzmoser2012CorrelationAnalysisCompositional>
#block[
Frauscher, Birgit, Nicolas von Ellenrieder, Rina Zelmann, et al. 2018. “Atlas of the Normal Intracranial Electroencephalogram: Neurophysiological Awake Activity in Different Cortical Areas.” #emph[Brain] 141 (4): 1130--44. #link("https://doi.org/10.1093/brain/awy035").

] <ref-frauscher2018AtlasNormalIntracranial>
] <refs>



