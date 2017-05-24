#include <stdio.h>
#include <map>
#include <assert.h>
#include <opencv2/core.hpp>
#include "drawingFunctions.h"

using namespace cv;
using std::map;


int main(int argc, char** argv)
{
	int K;
	printf("Enter number of clusters: ");
	scanf("%d", &K);
	char* files[] = { "dataset1.yml", "dataset2.yml", "dataset3.yml", "dataset4.yml" };
	int nFiles = 4;

	for (int i = 0; i < nFiles; i++) {

		const char * dataFileName = files[i];
		FileStorage fs;
		fs.open(dataFileName, FileStorage::READ);
		assert(fs.isOpened());
		Mat points;
		fs["points"] >> points;
		assert(points.empty() == false);

		//int K = nClusters;
		assert(K > 0);

		Mat labels;
		Mat centers;

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
		namedWindow(files[i]);
		imshow(files[i], img);
	}
	waitKey();
	destroyAllWindows();

	return 0;
}