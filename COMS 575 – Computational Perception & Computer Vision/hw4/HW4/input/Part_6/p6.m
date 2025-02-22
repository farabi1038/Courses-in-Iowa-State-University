fs = 44100;  % sampling frequency

% dial area code 319
dialAreaCode = dial_number('987', 0.5, 0.25, fs);
sound(dialAreaCode, fs);
audiowrite('319.wav', dialAreaCode, fs);

[Y,fs]=audioread('987.wav'); % read the WAV file
figure;
plot(1:2500,Y(1:2500), 'b*')     % plot the first 1000 samples
xlabel('Sample number (only the first 1000)');
ylabel('Y');
title('The wave for a dial tone area code 987');

figure; spectrogram(Y, 512, 256, 512, fs, 'yaxis');
title('Spectrogram for 471 area code');

S=spectrogram(Y, 512, 256, 512, fs, 'yaxis');

function [wave] = dial_number(dialedNumber, dialLength, pauseLength, samplingFreq)

    % Setup Pause Time
	pauseTime = 0 : 1/samplingFreq : pauseLength;
	pauseWave = sin(pauseTime);

	wave = 0;
	for i = 1 : length(dialedNumber)
		wave = [wave dial_digit(dialedNumber(i), dialLength, samplingFreq)];
		wave = [wave pauseWave];
	end
end

function [ dialedWave ] = dial_digit(dialKey, time, samplingFreq)
% Returns a wave of the dialed digit
% dialKey can be { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '#'}
% time can be (0, ... , infinite) in seconds

	% Setup Frequency Matrix
	freqRow = [697 770 852 941];
	freqCol = [1209 1336 1477];

	% Get Frequency Location of dialedNumber
	switch dialKey
		case '1'
			row = 1;
			col = 1;
		case '2'
			row = 1;
			col = 2;
		case '3'
			row = 1;
			col = 3;
		case '4'
			row = 2;
			col = 1;
		case '5'
			row = 2;
			col = 2;
		case '6'
			row = 2;
			col = 3;
		case '7'
			row = 3;
			col = 1;
		case '8'
			row = 3;
			col = 2;
		case '9'
			row = 3;
			col = 3;
		case '*'
			row = 4;
			col = 1;
		case '0'
			row = 4;
			col = 2;
		case '#'
			row = 4;
			col = 3;
		otherwise
			disp('Please enter a valid dial key.')
			return;
	end

	% Create time interval from 0 to time
	t = 0 : 1/samplingFreq : time;

	% Calculate corresponding row and column waves
	rowWave = sin(2 * pi * freqRow(row) * t);
	colWave = sin(2 * pi * freqCol(col) * t);

	% Calculate final wave
	dialedWave = (rowWave + colWave) / 2;
	return;
end
