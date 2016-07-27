#! /bin/bash

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# Check parameters...
if [ -z "$1" ]
  then
    echo "Need to provide a directory of *.line_graph files to learn from."
    exit 1
fi



# Keep learning models - each time main is called it tries a new random set of parameters...
while :
do
  timeout -s 9 8h python main.py $1 optimise
  sleep 10
done
