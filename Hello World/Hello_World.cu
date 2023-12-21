#include <iostream>

__global__ void helloWorld(){
    printf("Hello World CUDA");
}

int main(){
    helloWorld<<<1, 1>>>();
}