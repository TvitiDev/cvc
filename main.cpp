#include <cv.h>
#include <highgui.h>
#include <bits/unique_ptr.h>

using namespace cv;

bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, CvPoint & r);
double det (double a, double b, double c, double d);

void circles()
{
    IplImage* image = 0;
    // имя картинки задаётся первым параметром
    char* filename = const_cast<char *>("../picture/e46.jpg");
    // получаем картинку (в градациях серого)
    image = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);

    printf("[i] image: %s\n", filename);
    assert( image != 0 );

    // загрузим оригинальное изображении
    IplImage* src = cvLoadImage(filename );

    // хранилище памяти для кругов
    CvMemStorage* storage = cvCreateMemStorage(0);
    // сглаживаем изображение
    cvSmooth(image, image, CV_GAUSSIAN, 5, 5 );
    // поиск кругов
    CvSeq* results = cvHoughCircles(
            image,
            storage,
            CV_HOUGH_GRADIENT,
            10,
            200,
//            image->width/5,
            200,
            100,
            10,
            1000
    );
    // пробегаемся по кругам и рисуем их на оригинальном изображении
    for( int i = 0; i < results->total; i++ ) {
        float* p = (float*) cvGetSeqElem( results, i );
        CvPoint pt = cvPoint( cvRound( p[0] ), cvRound( p[1] ) );
        cvCircle( src, pt, cvRound( p[2] ), CV_RGB(0,0xff,0) );
    }

    // показываем
    cvNamedWindow( "cvHoughCircles", 1 );
    cvShowImage( "cvHoughCircles", src);

    // ждём нажатия клавиши
    cvWaitKey(0);

    // освобождаем ресурсы
    cvReleaseMemStorage(&storage);
    cvReleaseImage(& image);
    cvReleaseImage(&src);
    cvDestroyAllWindows();
}

int lines()
{
    struct customRect {
        int sideOne;
        int sideTwo;

        customRect(int a, int b) {
            sideOne = a;
            sideTwo = b;
        }
    };

    IplImage* src = 0;
    IplImage* dst=0;
    IplImage* color_dst=0;

    // имя картинки задаётся первым параметром
    char* filename = const_cast<char *>("../picture/777.jpg");
    // получаем картинку (в градациях серого)
    src = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);

    if( !src ){
        printf("[!] Error: cant load image: %s \n", filename);
        return -1;
    }

    printf("[i] image: %s\n", filename);

    // хранилище памяти для хранения найденных линий
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* lines = 0;
    int i = 0;

    dst = cvCreateImage( cvGetSize(src), 8, 1 );
    color_dst = cvCreateImage( cvGetSize(src), 8, 3 );

    cvSmooth(src, src, CV_GAUSSIAN, 7, 7);

    // детектирование границ
    cvCanny( src, dst, 50, 200, 3 );

    // конвертируем в цветное изображение
    cvCvtColor( dst, color_dst, CV_GRAY2BGR );

    // нахождение линий
    lines = cvHoughLines2( dst, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/120, 25, 20, 10 );

    // нарисуем найденные линии
//    for( i = 0; i < lines->total; i++ ){
//        CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
//        cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 1, CV_AA, 0 );
//    }

    std::vector < std::vector <int> > linesGroup(46);

    // обходим все линии и группируем их
    for( i = 0; i < lines->total; i++ ){
        CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);

        int x = line[1].x - line[0].x;
        int y = line[1].y - line[0].y;

        double grad = (180 / CV_PI) * acos(y / sqrt(x * x + y * y));

        // номер группы (целая часть от числа)
        int numberGroup = (int) trunc(grad / 4);

        // помещаем в группу номер линии
        linesGroup.at(numberGroup).push_back(i);
    }

    // заходим в группу и полным перебором проверяем отрезки, что бы дистанция между
    // ними была не более определенного расстояния. Если это так - добавляем потенциальную
    // пару

    // вектор под хранение предполагаемых параллелипипидов
    std::vector < std::vector <customRect> > rectGroup(46);

    for (int group = 0; group < linesGroup.size(); group++) {
        std::vector <int> subGroup = linesGroup.at(group);
        int subGroupSize = subGroup.size();

        printf("group %d, subGroup %d \n", group, subGroup.size());

        for (int j = 0; j < subGroupSize - 1; j++) {
            CvPoint* line1 = (CvPoint*)cvGetSeqElem(lines,subGroup.at(j));

            for (int k = j + 1; k < subGroupSize; k++) {
                CvPoint* line2 = (CvPoint*)cvGetSeqElem(lines,subGroup.at(k));

                int c1x = (line1[0].x + line1[1].x) / 2;
                int c1y = (line1[0].y + line1[1].y) / 2;
                int c2x = (line2[0].x + line2[1].x) / 2;
                int c2y = (line2[0].y + line2[1].y) / 2;

                double dist = sqrt( pow(c2x - c1x, 2) + pow(c2y - c1y, 2) );

                // максимальная дистанция между векторами (их серединами)
                if (dist > 25 && dist < 100) {
                    customRect cR = {subGroup.at(k), subGroup.at(j)};
                    rectGroup.at(group).push_back(cR);
                }
            }
        }
    }
    
    // Теперь перебираем пары отрезков на возможное пересечение. При проверке на пересечение продлеваем отрезок в обе 
    // стороны, что бы нивелировать погрешность выделения границ

    for (int group = 0; group < rectGroup.size(); group++) {
        // элементы группы
        std::vector <customRect> subGroup = rectGroup.at(group);
        int subGroupSize = subGroup.size();
        // сравниваем его со всеми парами из групп
        for (int nextGroup = group; nextGroup < rectGroup.size(); nextGroup++) {
            // соседние группы не берем, что бы не были слишком острые углы
            if (nextGroup > group - 2 && nextGroup < group + 2) {
                continue;
            }
            std::vector <customRect> nextSubGroup = rectGroup.at(nextGroup);
            int nextSubGroupSize  = nextSubGroup.size();
            for (int j = 0; j < subGroupSize; j++) {
                for (int k = 0; k < nextSubGroupSize; k++) {
                    // сравниваем попарно отрезки из двух групп
                    customRect cr1 = subGroup.at(j);
                    customRect cr2 = nextSubGroup.at(k);
                    CvPoint* line1 = (CvPoint*)cvGetSeqElem(lines, cr1.sideOne);
                    CvPoint* line2 = (CvPoint*)cvGetSeqElem(lines, cr1.sideTwo);
                    CvPoint* line3 = (CvPoint*)cvGetSeqElem(lines, cr2.sideOne);
                    CvPoint* line4 = (CvPoint*)cvGetSeqElem(lines, cr2.sideTwo);
                    CvPoint rect[4];

                    // проверяем, есть ли пересечения
                    if (
                        intersection(line1[0], line1[1], line3[0], line3[1], rect[0])
                        &&
                        intersection(line1[0], line1[1], line4[0], line4[1], rect[1])
                        &&
                        intersection(line2[0], line2[1], line3[0], line3[1], rect[3])
                        &&
                        intersection(line2[0], line2[1], line4[0], line4[1], rect[2])
                    ) {
                        // точки переречения
                        cvCircle(color_dst, rect[0], 2, CV_RGB(255,0,0));
                        cvCircle(color_dst, rect[1], 2, CV_RGB(255,0,0));
                        cvCircle(color_dst, rect[2], 2, CV_RGB(255,0,0));
                        cvCircle(color_dst, rect[3], 2, CV_RGB(255,0,0));

                        // четырехугольник по точкам пересечения
                        CvPoint* ppt[1] = { rect };
                        int npt[1] = { 4 };
                        cvPolyLine(color_dst, ppt, npt, 1, 1, CV_RGB(255,0,0), 3, 0);

                        // просто линии, которые пересекались
//                        cvLine( color_dst, line1[0], line1[1], CV_RGB(255,0,0), 2, CV_AA, 0 );
//                        cvLine( color_dst, line2[0], line2[1], CV_RGB(0,255,0), 2, CV_AA, 0 );
//                        cvLine( color_dst, line3[0], line3[1], CV_RGB(0,0,255), 2, CV_AA, 0 );
//                        cvLine( color_dst, line4[0], line4[1], CV_RGB(100,255,100), 2, CV_AA, 0 );
                    }
                }
            }
        }
    }

    printf("lines->total  %d \n", lines->total);

    // показываем
    cvNamedWindow( "Source", 1 );
    cvShowImage( "Source", src );

    cvNamedWindow( "Hough", 1 );
    cvShowImage( "Hough", color_dst );

    // ждём нажатия клавиши
    cvWaitKey(0);

    cvSaveImage("../picture/out3.jpg", color_dst);

    // освобождаем ресурсы
    cvReleaseMemStorage(&storage);
    cvReleaseImage(&src);
    cvReleaseImage(&dst);
    cvReleaseImage(&color_dst);
    cvDestroyAllWindows();

}

// делаем так, что бы точка А была левой нижней, а B была правой верхней
void rotateLine(Point2f a, Point2f b, Point2f &outA, Point2f &outB);

// поиск пересечения двух прямых
bool intersection(Point2f A, Point2f B, Point2f C, Point2f D, CvPoint &p) {
    double k = 36;

    Point2f a;
    Point2f b;
    Point2f c;
    Point2f d;

    rotateLine(A, B, a, b);
    rotateLine(C, D, c, d);

    int A1 = a.y - b.y;
    int B1 = b.x - a.x;
    int C1 = a.x * b.y - b.x * a.y;

    int A2 = c.y - d.y;
    int B2 = d.x - c.x;
    int C2 = c.x * d.y - d.x * c.y;

    double zn = det (A1, B1, A2, B2);
    if (abs (zn) < 1e-8) {
        return false;
    } else {
        int X = - det (C1, B1, C2, B2) / zn;
        int Y = - det (A1, C1, A2, C2) / zn;

        if (X >= a.x - k && X <= b.x + k && X >= c.x - k && X <= d.x + k
            &&
            Y >= a.y - k && Y <= b.y + k && Y >= c.y - k && Y <= d.y + k) {
            p.x = X;
            p.y = Y;
            return true;
        } else {
            return false;
        }
    }
}

double det(double a, double b, double c, double d) {
    return a * d - b * c;
}

void rotateLine(Point2f a, Point2f b, Point2f &outA, Point2f &outB) {
    if (a.y < a.y) {
        outA = a;
        outB = b;
    } else {
        outA = b;
        outB = a;
    }

    if (a.x < b.x) {
        outA = a;
        outB = b;
    } else {
        outA = b;
        outB = a;
    }
}

int main() {

//    circles();
    lines();

    return 0;
}