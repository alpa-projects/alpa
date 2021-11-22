#!/bin/bash
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.44.28 "killall python3" &
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.38.88 "killall python3" &
ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/berkeley-aws-oregon.pem ubuntu@172.31.43.214 "killall python3" &
ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs sudo kill -9
