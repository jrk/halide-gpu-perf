//
//  ViewController.h
//  Metal iOS Perf
//
//  Created by Jonathan Ragan-Kelley on 1/10/17.
//  Copyright © 2017 Halide. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface ViewController : UIViewController {
    IBOutlet UILabel *appLabel;
    IBOutlet UILabel *metalRuntimeLabel;
    IBOutlet UILabel *cpuRuntimeLabel;
}

-(void)setApp:(NSString*)app;
-(void)setRuntime:(float)ms forMetal:(BOOL)metal;

@end
