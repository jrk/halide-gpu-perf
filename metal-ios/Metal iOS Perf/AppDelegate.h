//
//  AppDelegate.h
//  Metal iOS Perf
//
//  Created by Jonathan Ragan-Kelley on 1/10/17.
//  Copyright © 2017 Halide. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <Metal/Metal.h>

@interface AppDelegate : UIResponder <UIApplicationDelegate>

@property (strong, nonatomic) UIWindow *window;

@property (strong, nonatomic) id<MTLDevice> device;
@property (strong, nonatomic) id<MTLCommandQueue> commandQueue;

@end

