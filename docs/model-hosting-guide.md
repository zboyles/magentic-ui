# Model Hosting Guide

MagenticLite talks to models through servers that expose an **OpenAI-compatible `/v1/chat/completions` endpoint**. To connect one, provide its base URL, typically ending in `/v1`. You have a few ways to host one:

- **Hugging Face Inference Endpoints.** Managed GPU hosting, billed per minute. Walked through in [Option A](#option-a-hugging-face-inference-endpoints) below.
- **Bring your own GPU.** Run the models yourself with vLLM on GPU machines you manage. Walked through in [Option B](#option-b-bring-your-own-gpu-with-vllm) below.
- **Microsoft Foundry Managed Compute.** Managed GPU hosting on Azure, billed per hour. Walked through in [Option C](#option-c-microsoft-foundry-managed-compute) below.
- **Apple Silicon (MLX).** On-device MagenticBrain + Fara via `mlx_lm.server` / `mlx_vlm.server`. Walked through in [Option D](#option-d-apple-silicon-with-mlx) below.

Whichever path you pick, you end up with the same three values to paste into MagenticLite's onboarding (or into **Settings → Models**): an OpenAI-compatible URL, a model name, and an API key.

> **Each model needs its own endpoint.** MagenticLite uses one model for the orchestrator role and another for browser use. On a managed platform, that means **two deployments**, each with its own URL and key. With your own GPUs, you run two vLLM servers; this guide uses two single-GPU machines.

---

## Option A: Hugging Face Inference Endpoints

### Prerequisites

- A [Hugging Face account](https://huggingface.co/join) with a payment method or pre-paid credits.
- A Hugging Face access token from [Settings → Access Tokens](https://huggingface.co/settings/tokens). Choose **Fine-grained** and grant at least:
  - _Inference → Make calls to your Inference Endpoints_
  - _Repositories → Read access to contents of all public gated repos you can access_

  Save the token — it's shown only once.

### A1. Deploy the model

You'll repeat this once per model role you want to use (browser use and/or orchestrator).

1. Open the model card on Hugging Face — [`microsoft/Fara1.5-9B`](https://huggingface.co/microsoft/Fara1.5-9B) for browser use or [`microsoft/MagenticBrain`](https://huggingface.co/microsoft/MagenticBrain) for orchestration — and click **Deploy → Inference Endpoints (dedicated)**.

2. The first time you do this, Hugging Face shows an OAuth consent screen for the _Inference Endpoints_ app. Click **Authorize**. After authorization, the create-endpoint form **resets to defaults** — re-apply your settings before deploying.

3. Configure the endpoint:

   | Field                       | Value                                                            |
   | --------------------------- | ---------------------------------------------------------------- |
   | Endpoint Name               | e.g. `fara-15-9b-magentic-lite` or `magenticbrain-magentic-lite` |
   | Hardware                    | the form's current **Suggested** configuration                   |
   | Inference Engine            | **vLLM**                                                         |
   | Authentication              | **Private (default)**                                            |
   | Autoscaling → Scale-to-Zero | **15 minutes**                                                   |

   When validated, Hugging Face suggested:
   - **Fara:** Nvidia L40S ×1 (48 GB, ~$1.80/hr)
   - **MagenticBrain:** Nvidia RTX PRO 6000 Blackwell ×1 (96 GB, ~$2.75/hr)

   In **Advanced Configuration**, enter the complete string for your model in the single **Container Arguments** text field:

   - **Fara:** `--dtype bfloat16 --max-model-len 262144 --limit-mm-per-prompt {"image":10}`
   - **MagenticBrain:** `--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768`

4. Click **Create Endpoint**.

   The endpoint takes ~5–10 minutes to build (Hugging Face downloads the weights and starts vLLM). **Billing starts when the status changes from `Initializing` to `Running`**.

### A2. Connect MagenticLite

When the endpoint is **Running**, copy the **Endpoint URL** from its overview page and append `/v1` when entering it in MagenticLite. Use the URL Hugging Face provides for each deployment; its hostname depends on the cloud provider and region you selected.

Open MagenticLite and fill in the **Browser use model** card (and/or the **Orchestrator** card). On first launch this is part of the onboarding flow; if you've already onboarded, find the same fields under **Settings → Models**.

The endpoint name identifies the deployment and does not determine the Model Name. Use the model ID shown below.

| Field        | Browser use model (Fara)                  | Orchestrator model (MagenticBrain)                 |
| ------------ | ----------------------------------------- | -------------------------------------------------- |
| Endpoint URL | the Fara Endpoint URL with `/v1` appended | the MagenticBrain Endpoint URL with `/v1` appended |
| Model Name   | `microsoft/Fara1.5-9B`                    | `microsoft/MagenticBrain`                          |
| API Key      | your Hugging Face access token (`hf_…`)   | your Hugging Face access token (`hf_…`)            |

Click **Verify & Save**. See [Verification fails](#verification-fails) below if you hit an error.

### A3. Scale-to-zero and cold starts

If you enabled scale-to-zero in step A1, the endpoint **automatically scales to zero** after the configured idle window (15 minutes in this guide). It stops billing and is shown as `Scaled to zero` on the Hugging Face dashboard. This differs from manually pausing an endpoint, which requires a manual resume.

The next request to a scaled-to-zero endpoint triggers a **cold start**: Hugging Face brings a replica back up, which often takes **30–90 seconds** and can take several minutes for a larger model. During this window MagenticLite may show an error like _"Endpoint returned HTTP 503"_ on **Verify & Save**, or the first chat turn may appear to hang. Wait and retry.

Subsequent requests respond at normal speed until the endpoint scales to zero again.

---

## Option B: Bring Your Own GPU with vLLM

To host models with [vLLM](https://docs.vllm.ai/), you need to provide your own GPU-equipped Linux machines. This guide uses two single-GPU machines as the example: one for Fara and one for MagenticBrain. For the model-card commands below, plan for at least 48 GB of GPU memory for Fara and 96 GB for MagenticBrain. Hardware needs may change if you adjust the context length or other vLLM options.

### B1. Start the model servers

On the **Fara machine**, run the command from the model card:

```bash
vllm serve microsoft/Fara1.5-9B \
   --dtype bfloat16 \
   --max-model-len 262144 \
   --limit-mm-per-prompt image=10
```

On the **MagenticBrain machine**, run the command from the model card:

```bash
vllm serve microsoft/MagenticBrain \
   --enable-auto-tool-choice \
   --tool-call-parser hermes \
   --max-model-len 32768
```

### B2. Connect MagenticLite

The default model names are the repository IDs shown below. If you add `--served-model-name`, use that value instead.

| Field        | Browser use model (Fara)                 | Orchestrator model (MagenticBrain)       |
| ------------ | ---------------------------------------- | ---------------------------------------- |
| Endpoint URL | `http://<fara-host>:8000/v1`             | `http://<brain-host>:8000/v1`            |
| Model Name   | `microsoft/Fara1.5-9B`                   | `microsoft/MagenticBrain`                |
| API Key      | empty unless the server uses `--api-key` | empty unless the server uses `--api-key` |

Click **Verify & Save**.

---

## Option C: Microsoft Foundry Managed Compute

### Prerequisites

- An [Azure subscription](https://azure.microsoft.com/free/) with a valid payment method. Free and trial subscriptions don't work for GPU deployments.
- A **hub-based** project in Foundry. The newer "Foundry project" type does not support Managed Compute. If you don't have one, create it from the [Foundry portal](https://ai.azure.com/) under **+ New project → Hub-based project**. Pick a region with H100 or A100 inventory (East US 2 and Sweden Central are good defaults).
- Quota for enough dedicated vCPUs in the chosen region. **Standard_NC24ads_A100_v4** is a good VM SKU for both [Fara1.5-9B](https://aka.ms/fara-foundry) and [MagenticBrain](https://aka.ms/MagenticBrain-foundry) for testing and typical single-user use, and each instance consumes 24 vCPUs from the quota family. In [Azure Quotas](https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/overview), select **Machine learning**, then request **Standard NCADSA100v4 Family Cluster Dedicated vCPUs** in the same region as your Foundry project. For the usual two-deployment setup with Fara and MagenticBrain running concurrently at instance count 1, request 48 dedicated vCPUs. Larger A100 or H100 SKUs also work if you want extra headroom or have them readily available, but they cost more. Approval can take 24–48 hours.

### C1. Deploy the model

You'll repeat this once per model role you want to use (browser use and/or orchestrator).

1. Open the model card in [Foundry Explore models](https://ai.azure.com/explore/models): [Fara1.5-9B](https://aka.ms/fara-foundry) for browser use or [MagenticBrain-14B](https://aka.ms/MagenticBrain-foundry) for orchestration.

2. On the model card, click **Use this model**. If Foundry asks you to select a project, choose an existing hub-based project or create a new one. For a new project, keep the default hub unless you already have a shared hub for this work, and pick a region with GPU inventory such as East US 2 or Sweden Central.

   If project creation fails with a `Microsoft.Resources/subscriptions/resourcegroups/write` authorization error, your account can see the model but cannot create the Azure resource group behind the Foundry project. Use an existing project where you have access, or ask the subscription owner to grant you a role such as Contributor on the subscription or target resource group, then refresh your credentials and try again.

3. Continue to the deployment wizard. If you're presented with purchase options, pick **Managed Compute**.

4. Configure the deployment:

   | Field           | Value                                                                                                                       |
   | --------------- | --------------------------------------------------------------------------------------------------------------------------- |
   | Endpoint name   | anything, e.g. `fara-15-9b-magentic-lite`. Becomes part of the URL.                                                         |
   | Deployment name | anything, e.g. `fara1-5-9b-1` or `magenticbrain-14b-1`. This is for tracking the deployment in Foundry.                     |
   | Virtual machine | **Standard_NC24ads_A100_v4**. Larger A100 or H100 SKUs also work, but they are usually unnecessary for testing.             |
   | Instance count  | **1**. Foundry may default to 3 instances; reduce it to 1 for testing or typical single-user use to avoid unnecessary cost. |

   Both Fara and MagenticBrain are served by vLLM under the hood, so the deployed endpoint exposes a fully OpenAI-compatible `/v1/chat/completions` route — text and vision-language requests both work.

5. Click **Deploy**.

   Provisioning takes ~15–20 minutes per model: Foundry allocates the VM, pulls the container, and warms up vLLM. **Billing starts when the VM is allocated**, not when the endpoint reaches `Healthy`.

### C2. Connect MagenticLite

For each deployment, open **Models + endpoints** in your Foundry project and click into the deployment:

- **REST endpoint** (Details tab): copy the endpoint through `/v1`, for example `https://<endpoint-name>.<region>.inference.ml.azure.com/v1`.
- **Model ID** (deployment details): use the model name segment after `/models/`, for example `Fara1.5-9B` or `MagenticBrain-14B`. Do not use the deployment name here.
- **Primary key** (Consume tab): the API key Foundry generated for that endpoint.

Open MagenticLite and fill in the **Browser use model** card (and/or the **Orchestrator** card). On first launch this is part of the onboarding flow; if you've already onboarded, find the same fields under **Settings → Models**.

| Field        | Browser use model (Fara)                                     | Orchestrator model (MagenticBrain)                            |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------- |
| Endpoint URL | `https://<fara-endpoint>.<region>.inference.ml.azure.com/v1` | `https://<brain-endpoint>.<region>.inference.ml.azure.com/v1` |
| Model Name   | `Fara1.5-9B`                                                 | `MagenticBrain-14B`                                           |
| API Key      | the primary key from the Fara endpoint's Consume tab         | the primary key from the MagenticBrain endpoint's Consume tab |

Click **Verify & Save**. See [Verification fails](#verification-fails) below if you hit an error.

### C3. Idle behavior and cost

Foundry Managed Compute deployments **do not scale to zero**. The VM stays allocated and billed by the hour for as long as the deployment exists, whether or not traffic is flowing. An A100 deployment in East US 2 runs roughly $3–4 per hour at list price (H100 is roughly twice that); check the [Azure VM pricing page](https://azure.microsoft.com/pricing/details/virtual-machines/linux/) for current rates in your region. Multiply by the number of deployments you keep running.

To stop the meter, **delete the deployment** from the **Models + endpoints** page. Redeploying from the catalog later takes the same ~15–20 minutes.

---

## Option D: Apple Silicon with MLX

On an Apple Silicon Mac you can serve the MagenticLite models locally with [MLX](https://github.com/ml-explore/mlx). MagenticLite still talks HTTP OpenAI Chat Completions — it does not load MLX weights itself. This path starts two small servers that expose `/v1/chat/completions`.

This is **not** a replacement for the vLLM / Foundry options above. `mlx_lm.server` is a basic development server, not a production inference stack.

### Prerequisites

- macOS on Apple Silicon (the extra is skipped on other platforms).
- MagenticLite already installed in a venv as in the [Installation](./installation.md) guide.
- Enough unified memory to hold **both** models plus MagenticLite and the Quicksand VM. Roughly **32 GB is tight**; **64 GB+ is comfortable**.

### D1. Install the MLX extra

From the same venv you use to run `magentic-ui`:

```bash
uv pip install "magentic_ui[mlx]"
```

That extra pulls in `mlx-lm` (MagenticBrain) and `mlx-vlm` (Fara). Fara is a vision-language model; `mlx-lm` alone is not enough.

### D2. Start the servers

```bash
magentic-ui mlx-serve
```

This starts two processes and prints the values to paste into MagenticLite:

| Role | Default URL | Default model name |
|------|-------------|--------------------|
| Orchestrator | `http://127.0.0.1:8100/v1` | `mlx-community/MagenticBrain-8bit` |
| Browser use | `http://127.0.0.1:8101/v1` | `mlx-community/Fara1.5-9B-8bit` |

Useful flags: `--role brain` / `--role fara`, `--brain-port`, `--fara-port`, `--brain-model`, `--fara-model`. Weights download into the Hugging Face cache on first launch.

Leave this terminal running. In another terminal, start MagenticLite as usual (`magentic-ui --port 8081`).

### D3. Connect MagenticLite

Open MagenticLite and fill in **Settings → Models** (or first-run onboarding):

| Field        | Browser use model (Fara)                    | Orchestrator model (MagenticBrain)                 |
| ------------ | ------------------------------------------- | -------------------------------------------------- |
| Endpoint URL | `http://127.0.0.1:8101/v1`                  | `http://127.0.0.1:8100/v1`                         |
| Model Name   | `mlx-community/Fara1.5-9B-8bit`             | `mlx-community/MagenticBrain-8bit`                 |
| API Key      | `not-needed`                                | `not-needed`                                       |

Click **Verify & Save**. Verification only probes `GET /v1/models`; it does not exercise tool calling or vision.

A commented YAML snippet lives in [`config.yaml.example`](../config.yaml.example).

### D4. Caveats

- Two resident 8-bit models plus the browser VM need a lot of unified memory. If the Mac is tight, run `--role brain` or `--role fara` and MagenticLite's `omniagent_only` / `websurfer_only` agent mode.
- `mlx_lm.server` may put MagenticBrain tool calls in the OpenAI-native `tool_calls` field instead of leaving `<tool_call>` XML in `message.content`. OmniAgent currently parses tools from content; if the orchestrator *talks about* `delegate_cua` but never runs it, that is this mismatch — not a down Fara server.
- Community MLX quants are sibling weights of `microsoft/MagenticBrain` and `microsoft/Fara1.5-9B`. Quality and tool/vision fidelity can differ from the documented vLLM deployments.

---

## Verification fails

When you click **Verify & Save**, MagenticLite sends a probe request to the endpoint:

- During onboarding, a successful verification finishes the onboarding flow and sends you to the sample-tasks page.
- In Settings, the button updates to **Connection Verified** (with a check icon) once the endpoint responds.

If verification fails, the banner usually pinpoints the problem:

| Symptom (banner)                                                      | Likely cause                                                                                            |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `Endpoint returned HTTP 401` or `403`                                 | API Key field is empty or wrong (different endpoints return one or the other for the same problem)      |
| `Endpoint returned HTTP 503` on the first attempt                     | Cold start (Hugging Face Inference Endpoints only) — see [§A3](#a3-scale-to-zero-and-cold-starts) above |
| `Connection refused — is the server running?` or other network errors | Endpoint URL is wrong (typo in the host, missing `https://`, VPN/firewall issue)                        |
