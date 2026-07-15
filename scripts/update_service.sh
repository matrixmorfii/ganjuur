#!/bin/bash
cd /home/trinity/ganjuur

# Update the systemd service to point to longcat.py instead of gpt.py
sed -i 's|/home/trinity/ganjuur/gpt.py|/home/trinity/ganjuur/longcat.py|' /etc/systemd/system/ganjuur.service

# Reload systemd, restart the service
systemctl daemon-reload
systemctl restart ganjuur.service

# Wait and check
sleep 3
systemctl is-active ganjuur.service
