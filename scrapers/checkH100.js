const PROJECT_ID = "01ksc991j9nrp0gjdr9tpbzcyd";
const CLUSTER_ID = "lightning-lambda-prod";
const TOKEN = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZ3JpZCJdLCJleHAiOjE3ODE1MDI4OTEsImlhdCI6MTc4MDg5ODA5MSwiaXNzIjoiaHR0cHM6Ly9saWdodG5pbmcuYWkiLCJqdGkiOiJlZmY5ZDUwNC1lMWMwLTQxZmYtOTI1Ni04ZTUzMWUxMmJjNTAiLCJuYmYiOjE3ODA4OTgwOTEsInN1YiI6IjBhZjcxYzNjLWFhNTAtNDhjZC04ZTViLTY1ZTVlZDZjMWZkMSIsInN1YmplY3RUeXBlIjoidXNlciJ9.FNJ6KFOCCjz1R_QrywODMNcOhdoncSkM3mzNBP393XbdaCZtXnF_AHZ49fXZL-6yNkxn0v74sA59ZdKTVreNZg";

let alreadyAvailable = false;

async function checkH100() {
    try {
        const res = await fetch(
            `https://lightning.ai/v1/projects/${PROJECT_ID}/clusters/${CLUSTER_ID}/accelerators?enabledOnly=true`,
            {
                headers: {
                    "accept": "*/*",
                    "authorization": `Bearer ${TOKEN}`
                }
            }
        );

        if (!res.ok) {
            console.log("Request failed:", res.status);
            return;
        }

        const data = await res.json();

        const accelerators =
            data.accelerator ||
            data.accelerators ||
            [];

        const h100 = accelerators.find(a =>
            a.provider === "LAMBDA_LABS" &&
            (
                a.family === "H100" ||
                a.instanceId === "gpu_1x_h100_sxm5" ||
                a.slug?.includes("h100")
            )
        );

        if (!h100) {
            console.log("H100 entry not found");
            return;
        }

        const available =
            h100.enabled === true &&
            h100.outOfCapacity === false;

        const now = new Date().toLocaleTimeString();

        if (available) {
            if (!alreadyAvailable) {
                console.log(JSON.stringify({ status: "available", time: now }, null, 2));
            }
            alreadyAvailable = true;
        } else {
            alreadyAvailable = false;
            console.log(JSON.stringify({ status: "unavailable", time: now }, null, 2));
        }
    } catch (err) {
        console.log("Error:", err);
    }
}
checkH100();