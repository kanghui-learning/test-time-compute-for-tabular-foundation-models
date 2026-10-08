# Third-party notices

This repository does not vendor third-party source code. It ships **patches** (in `patches/`)
against the projects below, plus a script (`scripts/install_forks.sh`) that downloads the
unmodified upstream sources, applies the patches locally and installs them. No model weights
(pretrained or fine-tuned) are distributed with this repository; each library downloads its
own weights from its publisher under the publisher's terms.

The patch files are derivative works of the respective upstream projects and are distributed
under those projects' licenses, listed below.

---

## TabPFN (Prior Labs)

- Upstream: https://github.com/PriorLabs/TabPFN (base: tag v8.0.7, commit `95ff6772`)
- Patch: `patches/tabpfn/`, `patches/tabpfn.diff` (modified files under `src/tabpfn/finetuning/`)
- Code license: **Prior Labs License v1.2 (Apache 2.0 with ADDITIONAL PROVISION)**, Dec 2025.
  Sections 1-9 are the Apache License 2.0; Section 10 is added. Full text:
  https://github.com/PriorLabs/TabPFN/blob/main/LICENSE (a copy is placed in
  `third_party/tabpfn/LICENSE` by the install script).

Section 10, quoted verbatim from the upstream `LICENSE`:

> 10. Additional attribution.
> If You distribute or make available the Work or any Derivative
> Work thereof relating to any part of the source or model weights,
> or a product or service (including another AI model) that contains
> any source or model weights, You shall (A) provide a copy of this
> License with any such materials; and (B) prominently display
> “Built with PriorLabs-TabPFN” on each related website, user interface, blogpost,
> about page, or product documentation. If You use the source or model
> weights or model outputs to create, train, fine tune, distil, or
> otherwise improve an AI model, which is distributed or made available,
> you shall also include “TabPFN” at the beginning of any such AI model name.
> To clarify, internal benchmarking and testing without external
> communication shall not qualify as distribution or making available
> pursuant to this Section 10 and no attribution under this Section 10
> shall be required.

What this requires of this repository (our patch is a Derivative Work relating to the source):

1. Ship a copy of the Prior Labs License with the patch (Section 10(A); also Apache 4(a)).
   -> `patches/tabpfn/LICENSE` must be present in the release.
2. Prominently display **“Built with PriorLabs-TabPFN”** in the repository README / project
   page and in any related website or blog post (Section 10(B)).
3. Modified files must carry prominent notices that they were changed (Apache 4(b)); the
   patch headers and `patches/README.md` serve this purpose.
4. If we ever distribute a *model* fine-tuned from TabPFN (e.g. a DiagScale delta or a
   fine-tuned checkpoint), its name must begin with “TabPFN” (Section 10, last sentence).
   This release distributes no such model.

TabPFN-3 model weights (`Prior-Labs/tabpfn_3`, used by default by `tabpfn` v8) are **not**
covered by the code license: they are under the **TABPFN-3 License v1.0** (non-commercial;
https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE). Users obtain them directly from
Prior Labs (the library asks the user to accept the license and uses a `TABPFN_TOKEN`). That
license allows distribution of Derivatives (e.g. fine-tuned weights) only for non-commercial
purposes, with the license copy, the attribution notice “The TABPFN-3 Model is licensed by
Prior Labs GmbH under the TABPFN-3 Non-Commercial License. Copyright © Prior Labs GmbH 2026.”,
a statement of modification, and no hosted/API service (Section 3). It also restricts using
Outputs to train a model competitive with TabPFN-3 (Section 2(d)).

## TabICL (Soda team, Inria)

- Upstream: https://github.com/soda-inria/tabicl (base: commit `46b91961` on `main`, between v2.1.1 and v2.2.0)
- Patch: `patches/tabicl/`, `patches/tabicl.diff`
- License: **BSD 3-Clause**, Copyright (c) 2025, Soda team @ Inria. Redistribution of the
  patch must retain the copyright notice, conditions and disclaimer
  (`patches/tabicl/LICENSE`); the Inria name may not be used to endorse derived products.
- Weights: `jingang/TabICL` on the Hugging Face Hub, tagged `bsd-3-clause`.

## TabFM (Google LLC)

- Upstream: https://github.com/google-research/tabfm (base: commit `2fb67e86` on `main`)
- Patch: `patches/tabfm/`, `patches/tabfm.diff`
- Code license: **Apache License 2.0**, Copyright 2026 Google LLC. Keep the license copy
  (`patches/tabfm/LICENSE`) and per-file headers; modified files are marked by the patch.
- Weights: `google/tabfm-1.0.0-pytorch` on the Hugging Face Hub. The model card states:
  “The model weights in this repository are released under the **TabFM Non-Commercial License
  v1.0** - see [LICENSE](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE).
  The source code is Apache 2.0 licensed via google-research/tabfm.” The weights license
  permits non-commercial research use only and, in Section 3(b), prohibits distributing the
  TabFM Model **or any Derivative** (fine-tuned / adapted versions included). This repository
  therefore must not ship TabFM weights or fine-tuned TabFM adapters/checkpoints.

## TabArena (AutoGluon)

- Upstream: https://github.com/autogluon/tabarena (pinned at commit `abd24c7f`, unmodified;
  `packages/tabarena` and `packages/bencheval` are installed)
- License: **Apache License 2.0**. TabArena pulls in AutoGluon (Apache-2.0) and other
  dependencies under their own licenses.

---

Built with PriorLabs-TabPFN.
