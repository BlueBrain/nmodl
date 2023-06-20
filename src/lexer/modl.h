/*
 * Copyright 2023 Blue Brain Project, EPFL.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * \file modl.h
 * \brief Legacy macro definitions from mod2c/nocmodl implementation
 *
 * Original implementation of NMODL use various flags to help
 * code generation. These flags are implemented as bit masks
 * which are later used during code printing. We are using ast
 * and hence don't need all bit masks.
 *
 * \todo Add these bit masks as enum-flags and remove this legacy header
 */

/// bit masks for block types where integration method are used
#define DERF  01000
#define KINF  02000
#define LINF  0200000
#define NLINF 04000
