#!/bin/bash
export MATLAB_HOME=/home/hugo/matlab/2017b/bin
export PACKAGE_JAVA_HOME=/opt/jdk1.7/
export DEPLOY_JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export MAVEN_HOME=/home/hugo

git pull
rm -rf driver/
export JAVA_HOME=$PACKAGE_JAVA_HOME
$MATLAB_HOME/deploytool -package driver.prj
export JAVA_HOME=$DEPLOY_JAVA_HOME
cd WorkflowBlock
mvn clean install
cd ..
