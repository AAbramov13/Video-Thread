#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

//создаем разные коннекты
    connect(&thread_1, &QThread::started, &threadObject_1, &ThreadObject::run);
    connect(&thread_2, &QThread::started, &threadObject_2, &ThreadObject::run);
    connect(&thread_3, &QThread::started, &threadObject_3, &StichingThread::run);
//коннект о разрушении
    connect(&threadObject_1, &ThreadObject::finished, &thread_1, &QThread::terminate);
    connect(&threadObject_2, &ThreadObject::finished, &thread_2, &QThread::terminate);
    connect(&threadObject_3, &StichingThread::finished, &thread_3, &QThread::terminate);
//тип посыл сигнала(ну тип заготовка)
    connect(&threadObject_1, &ThreadObject::sendImage, &threadObject_3, &StichingThread::setImage, Qt::DirectConnection);
    connect(&threadObject_2, &ThreadObject::sendImage, &threadObject_3, &StichingThread::setImage_2, Qt::DirectConnection);
//
    threadObject_1.moveToThread(&thread_1);
    threadObject_2.moveToThread(&thread_2);
    threadObject_3.moveToThread(&thread_3);
}

MainWindow::~MainWindow()
{
    delete ui;
}
//изменяем адреса
void MainWindow::on_read_cam1_clicked()
{
    threadObject_1.setMessage(ui->ip_cam1->text());
    threadObject_1.setFrame_name(ui->name_cam1->text());
}
//изменяем адреса
void MainWindow::on_read_cam2_clicked()
{
    threadObject_2.setMessage(ui->ip_cam2->text());
    threadObject_2.setFrame_name(ui->name_cam2->text());
}

void MainWindow::on_button_start_clicked()
{

    threadObject_1.setMessage(ui->ip_cam1->text());
    threadObject_1.setFrame_name(ui->name_cam1->text());

    threadObject_2.setMessage(ui->ip_cam2->text());
    threadObject_2.setFrame_name(ui->name_cam2->text());

    threadObject_1.setRunning(true);
    threadObject_2.setRunning(true);
    thread_1.start();
    thread_2.start();

}

void MainWindow::on_button_stop_clicked()
{
    threadObject_1.setRunning(false);
    threadObject_2.setRunning(false);
    threadObject_3.setRunning(false);

}

void MainWindow::on_button_start_2_clicked()
{

    threadObject_3.setRunning(true);
    thread_3.start();
}
