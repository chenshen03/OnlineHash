addpath(pwd);
addpath(fullfile(pwd, 'util'));
run vlfeat/toolbox/vl_setup
logInfo('done.');

cd online-hashing;

% CIFAR10
demo_online('OKH','cifar',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain',20000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)

demo_online('SketchHash','cifar',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain',20000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)

demo_online('AdaptHash','cifar',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain', 20000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)

demo_online('OSH','cifar',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain', 20000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)
	
demo_online('MIHash','cifar',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain',20000,'updateInterval',100,'trigger','mi','reservoirSize',1000,'override',1)

demo_online('HCOH','cifar',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain', 20000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)

demo_online('SDOH','cifar',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain', 20000,'updateInterval',2000,'trigger','fix','reservoirSize',200,'override',1)

% MNIST
demo_online('OKH','mnist',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain',20000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)

demo_online('SketchHash','mnist',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain',20000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)

demo_online('AdaptHash','mnist',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain', 20000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)

demo_online('OSH','mnist',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain', 20000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)
	
demo_online('MIHash','mnist',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain',20000,'updateInterval',100,'trigger','mi','reservoirSize',1000,'override',1)

demo_online('HCOH','mnist',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain', 20000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)

demo_online('SDOH','mnist',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain', 20000,'updateInterval',2000,'trigger','fix','reservoirSize',200,'override',1)

% PLACE
demo_online('OKH','places',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain',100000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)

demo_online('SketchHash','places',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain',100000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)

demo_online('AdaptHash','places',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain', 100000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)

demo_online('OSH','places',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain', 100000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)
	
demo_online('MIHash','places',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain',100000,'updateInterval',100,'trigger','mi','reservoirSize',5000,'override',1)

demo_online('HCOH','places',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain', 100000,'updateInterval',100,'trigger','fix','reservoirSize',200,'override',1)

demo_online('SDOH','places',64,'unseen',1,'ntrials',1,'ntest',2,'numTrain', 100000,'updateInterval',2000,'trigger','fix','reservoirSize',200,'override',1)
% LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab
