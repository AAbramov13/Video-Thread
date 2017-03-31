#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include "threadobject.h"
#include "stichingthread.h"
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_read_cam1_clicked();

    void on_read_cam2_clicked();

    void on_button_start_clicked();

    void on_button_stop_clicked();

    void on_button_start_2_clicked();

private:
    Ui::MainWindow *ui;
    QThread thread_1;
    QThread thread_2;
    QThread thread_3;

    ThreadObject threadObject_1;
    ThreadObject threadObject_2;
    StichingThread threadObject_3;

};

#endif // MAINWINDOW_H
