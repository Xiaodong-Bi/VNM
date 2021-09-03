
Blind image quality assessment by visual neuron matrix(VNM-NN)
=======================================================================

-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2021, Hua-wen Chang
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are 
met:

    * Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright 
      notice, this list of conditions and the following disclaimer in 
      the documentation and/or other materials provided with the distribution
      
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.

Plase use the citation provided below if it is useful to your research:

1）Hua-wen Chang, Xiao-Dong Bi and Chen Kai, "Blind image quality assessment by visual neuron matrix", 
	IEEE Signal Processing Letters
2）Hua-wen Chang, Xiao-Dong Bi and Chen Kai, "VNM_main", 
	URL:https://github.com/Xiaodong-Bi/VNM

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------%

=======================================================================
Author: Hua-wen Chang 
Version: 1.0  (June 6 2021)

The authors are with the School of Computer and Communication Engineering, Zhengzhou University of Light Industry, Zhengzhou 450002, China
Kindly report any suggestions or corrections to changhuawen@gmail.com


-----------------------------------------------------------
This package contains a Matlab implementation of the Blind image quality assessment by visual neuron matrix(VNM-NN).


Running on Matlab 

Input : A test image loaded in an array

Output: The quality scores are between 0 and 1, where 1 represents the same quality as the reference image.

Usage:
For quality evaluation, you can just run 'DemoTest.m' as follows:

1、Load the visual neuron matrix and network
	load('VNM.mat');  % load the feature detector (a matrix of size 128*768 generated by running TrainW(10000,16,16))
	load('net.mat'); % load the network

2、Load the image, for example
	Id = imread('1-1.png');

3、Extract the image features
	Features=VNM_NN(Id, VNM); %VNM represents the trained visual neuron matrix

4、Use the network to obtain the quality score
	Score  = sim(net,Features)  % net and B represent network and feature respectively

=======================================================================

-------------------------------------------------------------------------------------------------------------------------
Moreover, this package provides testing of the algorithm on CSIQ database.
You can download the database from:
CSIQ  http://vision.okstate.edu/?loc=csiq

1、Load database information and visual neuron matrix
load('CSIQ.mat');   % load database information and DMOS data  
load('VNM.mat');     % load the feature detector, Visual Neuron Matrix VNM

2、Extract image features from CSIQ database
3、Ada-boosting training network was used，the distorted images were divided into two parts with the 80%-20% train-test ratio
4、The final results show PLCC, SRCC, and KROCC results in the CSIQ database, as well as a scatter plot of the test images.

--------------------------------------------------------------------------------------------------------------------------
Contact
-------------------------
If you have any problems, questions, suggestions, or modifications, please contact me:
changhuawen@gmail.com



