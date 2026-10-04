%% Описание изображений простыми дескрипторами и распознавание объектов на основе сравнения с эталоном
% Наборы данных:
% - изображения букв английского алфавита notMNIST (kaggle.com);
% - изображения одежды Fashion MNIST PNG (kaggle.com).

%% 1. Загрузка изображений

% Получение доступа к файлам тренировочных изображений (закомментировать для тестирования)
imdDir = imageDatastore('C:\Users\gps.84\Downloads\archive\test',"IncludeSubfolders",true,"LabelSource","foldernames");

% Получение доступа к файлам тестовых изображений (раскомментировать для тестирования)
% imdDir = imageDatastore('C:\Users\gps.84\Downloads\archive\test',"IncludeSubfolders",true,"LabelSource","foldernames");

% 2. Вычисление дескрипторов

% Инициализация массивов для дескрипторов
D_1=zeros(length(imdDir.Files),2*28,'single');
D_2=zeros(length(imdDir.Files),28,'single');
D_3=D_2;
D_4=zeros(length(imdDir.Files),4,'single');
label =categorical([]);

% Вычисление дескрипторов для каждого изображения
tic
parfor i=1:length(imdDir.Files) 
    try

    % Загрузка изображения из файла    
    I=imread(imdDir.Files{i,1}); 

    % Пороговая бинаризация изображения
    I = imbinarize(I,'global'); 

    % Вычисление дескрипторов
    X=sum(I,1); % сумма значений пикселей по столбцам
    Y=sum(I,2); % сумма значений пикселей по строкам
    D_1(i,:)=[X Y']; % вариант дескриптора № 1
    D_2(i,:)=X-Y'; % вариант дескриптора № 2
    D_3(i,:)=abs(X-Y'); % вариант дескриптора № 3
    D_4(i,:)=[mean(D_1(i,:)) std(D_1(i,:)) mode(D_1(i,:)) median(D_1(i,:))]; % вариант дескриптора № 4

    % Метки классов набора
    label(i,1)=imdDir.Labels(i,1); 

    catch
        disp('Ошибка чтения файла!')
    end
end 
toc

%% Обучение моделей
classNames = categorical({'0'; '5'; '4'; '3'; '2'; '1'; '6'; '7'; '8'; '9'}, {'0' '5' '4' '3' '2' '1' '6' '7' '8' '9'});

% Обучение ансамбля K - ближайших соседей (KNN)
model_ensemble_knn = fitcensemble(D_1,label,'Method', 'Subspace', ...
    'NumLearningCycles', 30, ...
    'Learners', 'knn', 'ClassNames',classNames,'NPredToSample',28);

% Обучение ансамбля деревьев решений
model_ensemble_tree = fitcensemble(D_1,label,'Method', 'Bag', ...
    'NumLearningCycles', 30, ...
    'Learners', 'tree','ClassNames',classNames); 

% Обучение полносвязной нейронной сети 
model_neural_network = fitcnet(D_1, label, ...
    'LayerSizes', 100, ...
    'Activations', 'relu', ...
    'Lambda', 0, ...
    'IterationLimit', 1000, ...
    'Standardize',true,'ClassNames',classNames);

% Обучение машин опорных векторов (SVM)
template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
model_SVM = fitcecoc(D_1, label, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', classNames);

%% Распознавание тестовых изображений и оценка точности классификации

% Выбор дескриптора
DESCRIPTORS = D_1;

% Названия классификатора
name_model = ["Ансамбль ближайших соседей (KNN)", "Ансамбль деревьев решений (TREE)",...
    "Полносвязная нейронная сеть (NeuralNetwork)", "Машины опорных векторов классификации (SVС)"];

% Массив ячеек с моделями
models = {model_ensemble_knn, model_ensemble_tree, model_neural_network, model_SVM};

% Цикл для тестирование каждой модели
for i = 1:length(models)
    pred = predict(models{1,i}, DESCRIPTORS);
    figure, cm = confusionchart(label,pred);
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
    cm.Title = name_model(1,i);
    fprintf('Усредненное значение точности классификации изображений %s: %f \n',name_model(1,i), Acc)

end
