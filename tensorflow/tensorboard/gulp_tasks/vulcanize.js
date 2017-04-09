/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

var gulp = require('gulp');
var path = require('path');
var util = require('./util');
var vulcanize = require('gulp-vulcanize');
var replace = require('gulp-replace');
var rename = require('gulp-rename');
var header = require('gulp-header');

var HEADER_STR = '<!-- Copyright 2015 The TensorFlow Authors. All Rights Reserved.\n\
\n\
Licensed under the Apache License, Version 2.0 (the "License");\n\
you may not use this file except in compliance with the License.\n\
You may obtain a copy of the License at\n\
\n\
   http://www.apache.org/licenses/LICENSE-2.0\n\
\n\
Unless required by applicable law or agreed to in writing, software\n\
distributed under the License is distributed on an "AS IS" BASIS,\n\
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n\
See the License for the specific language governing permissions and\n\
limitations under the License.\n\
============================================================================\n\
\n\
This file is generated by `gulp` & `vulcanize`. Do not directly change it.\n\
Instead, use `gulp regenerate` to create a new version with your changes.\n\
-->\n\n'

var base = path.join(__dirname, '../components');
// List of redirects of the form path1|path2 for every tensorboard component
// in order to replace dashes with underscores.
// E.g. .../tf-tensorboard|.../tf_tensorboard
var redirects = util.tbComponents.map(function(dir) {
  return path.join(base, dir.replace(/_/g, '-')) + '|' + path.join(base, dir);
});

var nonTBComponents = util.getComponents(function(name) {
  var prefix = name.slice(0, 3);
  return prefix !== 'tf_'  && prefix !== 'vz_';
});

// These manual additions are necessary. The task should not inline these
// third-party javascript files. However, vulcanization still needs the HTML
// files found within those directories. Upon adding new third-party javascript,
// consider updating this list.
nonTBComponents.push('/tf-imports/d3.js');
nonTBComponents.push('/tf-imports/dagre.js');
nonTBComponents.push('/tf-imports/graphlib.js');
nonTBComponents.push('/tf-imports/lodash.js');
nonTBComponents.push('/tf-imports/plottable.js');

module.exports = function(overwrite) {
  return function() {
    var suffix = overwrite ? '' : '.OPENSOURCE';
    // Vulcanize TensorBoard without external libraries.
    gulp.src('components/tf_tensorboard/tf-tensorboard.html')
        .pipe(vulcanize({
          inlineScripts: true,
          inlineCss: true,
          stripComments: true,
          excludes: nonTBComponents,
          redirects: redirects
        }))
        .pipe(header(HEADER_STR))
        .pipe(rename('tf-tensorboard.html' + suffix))
        .pipe(gulp.dest('./dist'));
  }
}
