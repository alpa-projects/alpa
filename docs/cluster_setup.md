# AWS Cluster Setup Guide

1. Create a [placement group](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/placement-groups.html) on the AWS Management Console. Choose the `Cluster` placement strategy. This can make sure the interconnection bandwidth among different nodes in the cluster are high.
2. Create a securiy group on the AWS Management Console (EC2 -> Network & Security -> Security Groups).
3. Create an [EFS](https://console.aws.amazon.com/efs). This is used as an NFS for all nodes in the cluster. Please add the security group ID of the node you just started (can be found on the AWS Management Console) to the EFS to make sure your node can access the EFS. After that, you need to install the [efs-utils](https://docs.aws.amazon.com/efs/latest/ug/installing-other-distro.html) to mount the EFS on the node:
   ```bash
   git clone https://github.com/aws/efs-utils
   cd efs-utils
   ./build-deb.sh
   sudo apt-get -y install ./build/amazon-efs-utils*deb
   ```
   You can try to mount the EFS on the node by:
   ```bash
   mkdir -p ~/efs
   sudo mount -t efs {Your EFS file system ID}:/ ~/efs
   sudo chmod 777 ~/efs
   ```
   If this takes forever, make sure you configure the sercurity groups right.


Clone the git repos under `~/efs`.
