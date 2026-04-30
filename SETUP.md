## SETUP GUIDE DATAHUB (MACOS X)

Here's the complete steps:
1. Connect to VPN (full tunnel)
/opt/cisco/secureclient/bin/vpn connect vpn.ucsd.edu

Select option 0 (2-Step Secured - allthruucsd), enter credentials, approve Duo push.

2. SSH into DSMLP login node (Terminal 1)
ssh mbaghla@dsmlp-login.ucsd.edu


3. Check for existing pods and clean up
kubectl get pods
kubectl delete pod --all  # if anything is running


4. Launch A30 background pod (Terminal 1)
export K8S_TIMEOUT_SECONDS=43200 && launch.sh -v a30 -g 1 -c 8 -m 32 -b -W CSE151B_SP26_A00

Note the pod name from the output e.g. mbaghla-XXXXXXX

5. Start JupyterLab inside pod (Terminal 1)
kubesh mbaghla-XXXXXXX
jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password=''


6. Port-forward (Terminal 2 — new terminal on your Mac)
ssh mbaghla@dsmlp-login.ucsd.edu
kubectl port-forward mbaghla-XXXXXXX 8888:8888


7. SSH tunnel to your Mac (Terminal 3 — new terminal on your Mac)
ssh -L 8888:localhost:8888 mbaghla@dsmlp-login.ucsd.edu


8. Open JupyterLab in Chrome
http://localhost:8888/lab


Keep all 3 terminals open while working. If any terminal closes, the corresponding step breaks.
