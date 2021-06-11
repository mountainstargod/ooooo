#Install Dockstarter (Boiler Plate/ Template for Docker use)#

# Install DockStarter #
bash -c "$(curl -fsSL https://raw.githubusercontent.com/GhostWriters/DockSTARTer/master/main.sh)"


# Configure DS #

ds

# Run Compose #
ds -c up


# Install CTOP #


$ sudo wget https://github.com/bcicen/ctop/releases/download/v0.7.1/ctop-0.7.1-linux-amd64  -O /usr/local/bin/ctop
# Activate CTOP #
$ sudo chmod +x /usr/local/bin/ctop
