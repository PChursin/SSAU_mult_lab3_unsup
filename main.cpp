#include <stdio.h>
#include <map>
#include <assert.h>
#include <opencv2/core.hpp>
#include "drawingFunctions.h"

using namespace cv;
using std::map;


int main(int argc, char** argv)
{
	if (argc != 3) {
		argc = 3;
		argv = new char*[3];
		argv[1] = "dataset1.yml";
		argv[2] = "2";
	}
	//  аргументы командной строки:
	//  <имя файла с данными> <количество кластеров>
	assert(argc == 3);
	//  читаем данные из файла
	const char * dataFileName = argv[1];
	FileStorage fs;
	fs.open(dataFileName, FileStorage::READ);
	assert(fs.isOpened());
	Mat points;
	fs["points"] >> points;
	assert(points.empty() == false);

	int K = atoi(argv[2]);
	assert(K > 0);

	Mat labels;
	Mat centers;

	/* =================================================================== */
	//  Напишите код, выполняющий кластеризацию методом центров тяжести,
	//  используя функцию kmeans. В матрицу labels сохраните номера
	//  кластеров, к которым были отнесены точки выборки. В матрицу
	//  centers -- координаты центров, найденных кластеров.
	/* ------------------------------------------------------------------- */
	kmeans(points, K, labels, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
		10000, 0.001), 10, KMEANS_PP_CENTERS, centers);



		/* =================================================================== */

		// отображение номеров кластеров в BGR-цвета
		map<int, Scalar> clusterColors;
	clusterColors[0] = Scalar(255, 191, 0);
	clusterColors[1] = Scalar(0, 215, 255);
	clusterColors[2] = Scalar(71, 99, 255);
	clusterColors[3] = Scalar(0, 252, 124);
	clusterColors[4] = Scalar(240, 32, 160);

	// отображение распределения точек по кластерам
	// и центров кластеров
	Mat img(500, 500, CV_8UC3, Scalar(255, 255, 255));
	drawPoints(img, points, labels, getRanges(points), clusterColors, 0);
	drawPoints(img, centers, labels, getRanges(points), clusterColors, 2);
	namedWindow("clusters");
	imshow("clusters", img);
	waitKey();
	destroyAllWindows();

	return 0;
}