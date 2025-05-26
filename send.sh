#!/bin/bash
# Command to send to remote server polluks:

scp -r . polluks:/home/inf155986/prog_rownolegle/

ssh polluks "chmod -R 755 /home/inf155986/prog_rownolegle"