# Azure Container Apps deployment

Image: `ghcr.io/<owner>/audio_denoiser:latest` (built by `.github/workflows/ci.yml`).

## 1. One-time setup

```bash
az extension add --name containerapp --upgrade
az group create -n rg-denoiser -l westeurope
az containerapp env create -n cae-denoiser -g rg-denoiser
```

GHCR is private by default → create a pull secret so ACA can pull:

```bash
# GitHub PAT (classic) with read:packages, or a fine-grained token
az containerapp registry set -n denoiser -g rg-denoiser \
  --server ghcr.io --username <github-user> --password <PAT>
```

## 2. Create the app (2 vCPU / 4 GiB recommended for dns64 on CPU)

```bash
az containerapp create -n denoiser -g rg-denoiser --environment cae-denoiser \
  --image ghcr.io/<owner>/audio_denoiser:latest \
  --registry-server ghcr.io \
  --target-port 8000 --ingress external \
  --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 3 \
  --env-vars DENOISER_MODEL_NAME=dns48 DENOISER_TORCH_THREADS=2 \
             DENOISER_MAX_UPLOAD_MB=50 DENOISER_MAX_AUDIO_SECONDS=300
```

Scale-to-zero (`--min-replicas 0`) saves money but cold start ≈ 20–60s
(model load + image pull; slimmed image: CPU torch, no torchaudio,
lightweight static ffmpeg via imageio-ffmpeg, dns48 weights). For
demo-friendly latency use `--min-replicas 1`.

CI auto-updates the image on every `main` push once these repo secrets exist:
`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`
(federated credential for the workflow), `ACA_APP_NAME`, `ACA_RESOURCE_GROUP`.

## 3. File-upload strategy — what to use when

| Approach | How | Use when |
|---|---|---|
| **A. Ephemeral in-container (implemented)** | multipart POST → `/tmp/denoiser_jobs` → denoise → `FileResponse` → TTL sweep | MVP, files < 50 MB, no extra Azure resources. |
| **B. Azure Blob via SAS (recommended prod)** | frontend `PUT`s directly to Blob (SAS URL from backend) → backend job reads blob → writes output blob → returns SAS download URL | large files, persistence, multi-replica (ephemeral disk is per-replica!). |
| **C. Azure Files mount** | mount SMB share at `DENOISER_JOB_DIR` | you want a shared filesystem without changing code (slower than local SSD). |

Why not just keep A forever? ACA containers are **stateless/ephemeral**:
each replica has its own `/tmp`, restarts wipe it, and ingress times out
long CPU-bound requests (~30s+). B fixes all three: uploads bypass the
container, jobs can run async (queue + Event Grid), outputs survive restarts.

### Growing to B (no rewrite needed)
1. Add `azure-storage-blob` + `DENOISER_STORAGE_BACKEND=blob`,
   `DENOISER_BLOB_CONN` / managed-identity envs.
2. Add `POST /api/upload-url` (returns blob SAS) and make `/api/enhance`
   accept `{"blob": "<input-url>"}`; write result to `denoised/` prefix.
3. Frontend: `PUT` file → poll `GET /api/jobs/{id}` → download SAS URL.

Keep the current 50 MB / 5-min caps even on B — a 5-min clip on 2 vCPU
takes ~1–3 min (dns64 ≈ 1.0 RTF single-thread, ~0.6 with 4 threads).
For longer audio, chunk server-side or move to a GPU/aca-jobs worker.
