[Y,fs]=audioread('water.wav'); % read the WAV file
figure;
plot(1:1000,Y(1:1000), 'b*')     % plot the first 1000 samples
xlabel('Sample number (only the first 1000)');
ylabel('Y');
%title('The wave for animal sound');
%title('The wave for redstone sound');
title('The wave for water sound');

figure; spectrogram(Y, 512, 256, 512, fs, 'yaxis');
title('Spectrogram for the sound');

S=spectrogram(Y, 512, 256, 512, fs, 'yaxis');
sound(Y, fs)