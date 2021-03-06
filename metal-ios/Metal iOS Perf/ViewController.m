//
//  ViewController.m
//  Metal iOS Perf
//
//  Created by Jonathan Ragan-Kelley on 1/10/17.
//  Copyright © 2017 Halide. All rights reserved.
//

#import "ViewController.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

-(void)setApp:(NSString*)app {
    [appLabel setText:app];
}

-(void)setRuntime:(float)ms forMetal:(BOOL)metal {
    [(metal ? metalRuntimeLabel : cpuRuntimeLabel) setText:[NSString stringWithFormat:@"%.1f ms", ms]];
}

@end
