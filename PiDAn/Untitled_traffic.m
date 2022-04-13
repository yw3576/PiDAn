clear
%%
%genuine representations
path='...\experiment0401\rand_loc_False\eps_32\patch_size_5\trigger_14\num_poison_800\badnets_new\varying\feats\'

N = dir(fullfile(path,'*.mat'));
for i=1:43
    XX{i}=importdata(strcat(path,N(i).name));
end

%%
%poisoned representations in the target class labelled as class #3
load('...\experiment0401\rand_loc_False\eps_32\patch_size_5\trigger_14\num_poison_800\badnets_new\varying\features_2class_target')
load('...\experiment0401\rand_loc_False\eps_32\patch_size_5\trigger_14\num_poison_800\badnets_new\varying\features_2class_patched')
n1=1269;
n2=800;
A=repmat(features_target(1:n1,:),1,1);
C=repmat(features_patched(1:n2,:),1,1);

X=[A;C];
XX{24}=X;
%%
load('...\experiment0401\rand_loc_False\eps_32\patch_size_5\trigger_14\num_poison_800\badnets_new\varying\features_clean')

for i=1:43
A=XX{i};
X=[A];

% remove the means of representations based on the clean testing samples
% scale the representations to unit norm
X=(X-mean(features_clean{i},1));
for ii=1:size(X,1)
    X(ii,:)=X(ii,:)/norm(X(ii,:));
end

% get principal components by PCA, which consists of the latent subspace S1
% that is close to genuine subspace
Cx=cov(X);
[pm,l,g1]=pcacov(Cx);

% determine the parameter 'k', which is 'tt' in this code
sum=0;
for tt=1:512
    sum=sum+g1(tt);
    if sum>98
        break
    end
end

% coherence optimization to get the optimized weight vector a*, which is
% p(1,:) in this code
T=X;
Tm=T*pm(:,1:tt);
CT =cov(X');
CTm = cov(Tm');
[p,l,g]=pcacov(CT-CTm);

% likelihood ratio test
options=statset('MaxIter',1000);
for k=1:2
    gm = fitgmdist(p(:,1), k, 'Options',options);
    B(k)=gm.NegativeLogLikelihood;
%     B(k)=gm.BIC;
    gm.mu;
end
J(i)=(B(1)-B(2));
end

% anomaly index from APD
a=[J];
for i=1:43
    m(i)=median(abs(a(i)-a));
end
mm=1.1926*median(m);

ma = median(a);
b = abs(a-ma);
index = b/mm;
  
index1=index;
plot(index1)
index_target=index1(24);
 