# Streamlit deployment

The repository is ready for Streamlit Community Cloud. Deployment requires the repository owner to connect a GitHub account in the Streamlit workspace.

## Community Cloud settings

| Setting | Value |
|---|---|
| Repository | `ParthrChandurkar/predictive-network-route-optimizer` |
| Branch | `main` |
| Entry point | `app.py` |
| Python | `3.11` |

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app.
2. Connect the GitHub account that owns this repository.
3. Enter the settings above and deploy.
4. Copy the resulting `https://…streamlit.app` URL.
5. Set that URL as `NEXT_PUBLIC_STREAMLIT_URL` in the Vercel project and redeploy the web application.

The default dashboard generates a deterministic synthetic fallback in memory. The Kaggle LTE data and large trained LTE artifact are intentionally excluded from Git, so visitors must upload a competition CSV or reproduce the artifacts locally to use those panels.

Do not add Kaggle credentials or `.streamlit/secrets.toml` to Git.
