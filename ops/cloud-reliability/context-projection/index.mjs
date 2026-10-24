var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// work/context-cloud/node/runtime.mjs
import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { S3Client, GetObjectCommand, PutObjectCommand } from "@aws-sdk/client-s3";

// work/context-cloud/node/handler.mjs
import { timingSafeEqual } from "node:crypto";

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/node_modules/zod/v3/external.js
var external_exports = {};
__export(external_exports, {
  BRAND: () => BRAND,
  DIRTY: () => DIRTY,
  EMPTY_PATH: () => EMPTY_PATH,
  INVALID: () => INVALID,
  NEVER: () => NEVER,
  OK: () => OK,
  ParseStatus: () => ParseStatus,
  Schema: () => ZodType,
  ZodAny: () => ZodAny,
  ZodArray: () => ZodArray,
  ZodBigInt: () => ZodBigInt,
  ZodBoolean: () => ZodBoolean,
  ZodBranded: () => ZodBranded,
  ZodCatch: () => ZodCatch,
  ZodDate: () => ZodDate,
  ZodDefault: () => ZodDefault,
  ZodDiscriminatedUnion: () => ZodDiscriminatedUnion,
  ZodEffects: () => ZodEffects,
  ZodEnum: () => ZodEnum,
  ZodError: () => ZodError,
  ZodFirstPartyTypeKind: () => ZodFirstPartyTypeKind,
  ZodFunction: () => ZodFunction,
  ZodIntersection: () => ZodIntersection,
  ZodIssueCode: () => ZodIssueCode,
  ZodLazy: () => ZodLazy,
  ZodLiteral: () => ZodLiteral,
  ZodMap: () => ZodMap,
  ZodNaN: () => ZodNaN,
  ZodNativeEnum: () => ZodNativeEnum,
  ZodNever: () => ZodNever,
  ZodNull: () => ZodNull,
  ZodNullable: () => ZodNullable,
  ZodNumber: () => ZodNumber,
  ZodObject: () => ZodObject,
  ZodOptional: () => ZodOptional,
  ZodParsedType: () => ZodParsedType,
  ZodPipeline: () => ZodPipeline,
  ZodPromise: () => ZodPromise,
  ZodReadonly: () => ZodReadonly,
  ZodRecord: () => ZodRecord,
  ZodSchema: () => ZodType,
  ZodSet: () => ZodSet,
  ZodString: () => ZodString,
  ZodSymbol: () => ZodSymbol,
  ZodTransformer: () => ZodEffects,
  ZodTuple: () => ZodTuple,
  ZodType: () => ZodType,
  ZodUndefined: () => ZodUndefined,
  ZodUnion: () => ZodUnion,
  ZodUnknown: () => ZodUnknown,
  ZodVoid: () => ZodVoid,
  addIssueToContext: () => addIssueToContext,
  any: () => anyType,
  array: () => arrayType,
  bigint: () => bigIntType,
  boolean: () => booleanType,
  coerce: () => coerce,
  custom: () => custom,
  date: () => dateType,
  datetimeRegex: () => datetimeRegex,
  defaultErrorMap: () => en_default,
  discriminatedUnion: () => discriminatedUnionType,
  effect: () => effectsType,
  enum: () => enumType,
  function: () => functionType,
  getErrorMap: () => getErrorMap,
  getParsedType: () => getParsedType,
  instanceof: () => instanceOfType,
  intersection: () => intersectionType,
  isAborted: () => isAborted,
  isAsync: () => isAsync,
  isDirty: () => isDirty,
  isValid: () => isValid,
  late: () => late,
  lazy: () => lazyType,
  literal: () => literalType,
  makeIssue: () => makeIssue,
  map: () => mapType,
  nan: () => nanType,
  nativeEnum: () => nativeEnumType,
  never: () => neverType,
  null: () => nullType,
  nullable: () => nullableType,
  number: () => numberType,
  object: () => objectType,
  objectUtil: () => objectUtil,
  oboolean: () => oboolean,
  onumber: () => onumber,
  optional: () => optionalType,
  ostring: () => ostring,
  pipeline: () => pipelineType,
  preprocess: () => preprocessType,
  promise: () => promiseType,
  quotelessJson: () => quotelessJson,
  record: () => recordType,
  set: () => setType,
  setErrorMap: () => setErrorMap,
  strictObject: () => strictObjectType,
  string: () => stringType,
  symbol: () => symbolType,
  transformer: () => effectsType,
  tuple: () => tupleType,
  undefined: () => undefinedType,
  union: () => unionType,
  unknown: () => unknownType,
  util: () => util,
  void: () => voidType
});

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/node_modules/zod/v3/helpers/util.js
var util;
(function(util2) {
  util2.assertEqual = (_) => {
  };
  function assertIs(_arg) {
  }
  util2.assertIs = assertIs;
  function assertNever(_x) {
    throw new Error();
  }
  util2.assertNever = assertNever;
  util2.arrayToEnum = (items) => {
    const obj = {};
    for (const item of items) {
      obj[item] = item;
    }
    return obj;
  };
  util2.getValidEnumValues = (obj) => {
    const validKeys = util2.objectKeys(obj).filter((k) => typeof obj[obj[k]] !== "number");
    const filtered = {};
    for (const k of validKeys) {
      filtered[k] = obj[k];
    }
    return util2.objectValues(filtered);
  };
  util2.objectValues = (obj) => {
    return util2.objectKeys(obj).map(function(e) {
      return obj[e];
    });
  };
  util2.objectKeys = typeof Object.keys === "function" ? (obj) => Object.keys(obj) : (object) => {
    const keys = [];
    for (const key in object) {
      if (Object.prototype.hasOwnProperty.call(object, key)) {
        keys.push(key);
      }
    }
    return keys;
  };
  util2.find = (arr, checker) => {
    for (const item of arr) {
      if (checker(item))
        return item;
    }
    return void 0;
  };
  util2.isInteger = typeof Number.isInteger === "function" ? (val) => Number.isInteger(val) : (val) => typeof val === "number" && Number.isFinite(val) && Math.floor(val) === val;
  function joinValues(array, separator = " | ") {
    return array.map((val) => typeof val === "string" ? `'${val}'` : val).join(separator);
  }
  util2.joinValues = joinValues;
  util2.jsonStringifyReplacer = (_, value) => {
    if (typeof value === "bigint") {
      return value.toString();
    }
    return value;
  };
})(util || (util = {}));
var objectUtil;
(function(objectUtil2) {
  objectUtil2.mergeShapes = (first, second) => {
    return {
      ...first,
      ...second
      // second overwrites first
    };
  };
})(objectUtil || (objectUtil = {}));
var ZodParsedType = util.arrayToEnum([
  "string",
  "nan",
  "number",
  "integer",
  "float",
  "boolean",
  "date",
  "bigint",
  "symbol",
  "function",
  "undefined",
  "null",
  "array",
  "object",
  "unknown",
  "promise",
  "void",
  "never",
  "map",
  "set"
]);
var getParsedType = (data) => {
  const t = typeof data;
  switch (t) {
    case "undefined":
      return ZodParsedType.undefined;
    case "string":
      return ZodParsedType.string;
    case "number":
      return Number.isNaN(data) ? ZodParsedType.nan : ZodParsedType.number;
    case "boolean":
      return ZodParsedType.boolean;
    case "function":
      return ZodParsedType.function;
    case "bigint":
      return ZodParsedType.bigint;
    case "symbol":
      return ZodParsedType.symbol;
    case "object":
      if (Array.isArray(data)) {
        return ZodParsedType.array;
      }
      if (data === null) {
        return ZodParsedType.null;
      }
      if (data.then && typeof data.then === "function" && data.catch && typeof data.catch === "function") {
        return ZodParsedType.promise;
      }
      if (typeof Map !== "undefined" && data instanceof Map) {
        return ZodParsedType.map;
      }
      if (typeof Set !== "undefined" && data instanceof Set) {
        return ZodParsedType.set;
      }
      if (typeof Date !== "undefined" && data instanceof Date) {
        return ZodParsedType.date;
      }
      return ZodParsedType.object;
    default:
      return ZodParsedType.unknown;
  }
};

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/node_modules/zod/v3/ZodError.js
var ZodIssueCode = util.arrayToEnum([
  "invalid_type",
  "invalid_literal",
  "custom",
  "invalid_union",
  "invalid_union_discriminator",
  "invalid_enum_value",
  "unrecognized_keys",
  "invalid_arguments",
  "invalid_return_type",
  "invalid_date",
  "invalid_string",
  "too_small",
  "too_big",
  "invalid_intersection_types",
  "not_multiple_of",
  "not_finite"
]);
var quotelessJson = (obj) => {
  const json = JSON.stringify(obj, null, 2);
  return json.replace(/"([^"]+)":/g, "$1:");
};
var ZodError = class _ZodError extends Error {
  get errors() {
    return this.issues;
  }
  constructor(issues) {
    super();
    this.issues = [];
    this.addIssue = (sub) => {
      this.issues = [...this.issues, sub];
    };
    this.addIssues = (subs = []) => {
      this.issues = [...this.issues, ...subs];
    };
    const actualProto = new.target.prototype;
    if (Object.setPrototypeOf) {
      Object.setPrototypeOf(this, actualProto);
    } else {
      this.__proto__ = actualProto;
    }
    this.name = "ZodError";
    this.issues = issues;
  }
  format(_mapper) {
    const mapper = _mapper || function(issue) {
      return issue.message;
    };
    const fieldErrors = { _errors: [] };
    const processError = (error) => {
      for (const issue of error.issues) {
        if (issue.code === "invalid_union") {
          issue.unionErrors.map(processError);
        } else if (issue.code === "invalid_return_type") {
          processError(issue.returnTypeError);
        } else if (issue.code === "invalid_arguments") {
          processError(issue.argumentsError);
        } else if (issue.path.length === 0) {
          fieldErrors._errors.push(mapper(issue));
        } else {
          let curr = fieldErrors;
          let i = 0;
          while (i < issue.path.length) {
            const el = issue.path[i];
            const terminal = i === issue.path.length - 1;
            if (!terminal) {
              curr[el] = curr[el] || { _errors: [] };
            } else {
              curr[el] = curr[el] || { _errors: [] };
              curr[el]._errors.push(mapper(issue));
            }
            curr = curr[el];
            i++;
          }
        }
      }
    };
    processError(this);
    return fieldErrors;
  }
  static assert(value) {
    if (!(value instanceof _ZodError)) {
      throw new Error(`Not a ZodError: ${value}`);
    }
  }
  toString() {
    return this.message;
  }
  get message() {
    return JSON.stringify(this.issues, util.jsonStringifyReplacer, 2);
  }
  get isEmpty() {
    return this.issues.length === 0;
  }
  flatten(mapper = (issue) => issue.message) {
    const fieldErrors = {};
    const formErrors = [];
    for (const sub of this.issues) {
      if (sub.path.length > 0) {
        const firstEl = sub.path[0];
        fieldErrors[firstEl] = fieldErrors[firstEl] || [];
        fieldErrors[firstEl].push(mapper(sub));
      } else {
        formErrors.push(mapper(sub));
      }
    }
    return { formErrors, fieldErrors };
  }
  get formErrors() {
    return this.flatten();
  }
};
ZodError.create = (issues) => {
  const error = new ZodError(issues);
  return error;
};

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/node_modules/zod/v3/locales/en.js
var errorMap = (issue, _ctx) => {
  let message;
  switch (issue.code) {
    case ZodIssueCode.invalid_type:
      if (issue.received === ZodParsedType.undefined) {
        message = "Required";
      } else {
        message = `Expected ${issue.expected}, received ${issue.received}`;
      }
      break;
    case ZodIssueCode.invalid_literal:
      message = `Invalid literal value, expected ${JSON.stringify(issue.expected, util.jsonStringifyReplacer)}`;
      break;
    case ZodIssueCode.unrecognized_keys:
      message = `Unrecognized key(s) in object: ${util.joinValues(issue.keys, ", ")}`;
      break;
    case ZodIssueCode.invalid_union:
      message = `Invalid input`;
      break;
    case ZodIssueCode.invalid_union_discriminator:
      message = `Invalid discriminator value. Expected ${util.joinValues(issue.options)}`;
      break;
    case ZodIssueCode.invalid_enum_value:
      message = `Invalid enum value. Expected ${util.joinValues(issue.options)}, received '${issue.received}'`;
      break;
    case ZodIssueCode.invalid_arguments:
      message = `Invalid function arguments`;
      break;
    case ZodIssueCode.invalid_return_type:
      message = `Invalid function return type`;
      break;
    case ZodIssueCode.invalid_date:
      message = `Invalid date`;
      break;
    case ZodIssueCode.invalid_string:
      if (typeof issue.validation === "object") {
        if ("includes" in issue.validation) {
          message = `Invalid input: must include "${issue.validation.includes}"`;
          if (typeof issue.validation.position === "number") {
            message = `${message} at one or more positions greater than or equal to ${issue.validation.position}`;
          }
        } else if ("startsWith" in issue.validation) {
          message = `Invalid input: must start with "${issue.validation.startsWith}"`;
        } else if ("endsWith" in issue.validation) {
          message = `Invalid input: must end with "${issue.validation.endsWith}"`;
        } else {
          util.assertNever(issue.validation);
        }
      } else if (issue.validation !== "regex") {
        message = `Invalid ${issue.validation}`;
      } else {
        message = "Invalid";
      }
      break;
    case ZodIssueCode.too_small:
      if (issue.type === "array")
        message = `Array must contain ${issue.exact ? "exactly" : issue.inclusive ? `at least` : `more than`} ${issue.minimum} element(s)`;
      else if (issue.type === "string")
        message = `String must contain ${issue.exact ? "exactly" : issue.inclusive ? `at least` : `over`} ${issue.minimum} character(s)`;
      else if (issue.type === "number")
        message = `Number must be ${issue.exact ? `exactly equal to ` : issue.inclusive ? `greater than or equal to ` : `greater than `}${issue.minimum}`;
      else if (issue.type === "bigint")
        message = `Number must be ${issue.exact ? `exactly equal to ` : issue.inclusive ? `greater than or equal to ` : `greater than `}${issue.minimum}`;
      else if (issue.type === "date")
        message = `Date must be ${issue.exact ? `exactly equal to ` : issue.inclusive ? `greater than or equal to ` : `greater than `}${new Date(Number(issue.minimum))}`;
      else
        message = "Invalid input";
      break;
    case ZodIssueCode.too_big:
      if (issue.type === "array")
        message = `Array must contain ${issue.exact ? `exactly` : issue.inclusive ? `at most` : `less than`} ${issue.maximum} element(s)`;
      else if (issue.type === "string")
        message = `String must contain ${issue.exact ? `exactly` : issue.inclusive ? `at most` : `under`} ${issue.maximum} character(s)`;
      else if (issue.type === "number")
        message = `Number must be ${issue.exact ? `exactly` : issue.inclusive ? `less than or equal to` : `less than`} ${issue.maximum}`;
      else if (issue.type === "bigint")
        message = `BigInt must be ${issue.exact ? `exactly` : issue.inclusive ? `less than or equal to` : `less than`} ${issue.maximum}`;
      else if (issue.type === "date")
        message = `Date must be ${issue.exact ? `exactly` : issue.inclusive ? `smaller than or equal to` : `smaller than`} ${new Date(Number(issue.maximum))}`;
      else
        message = "Invalid input";
      break;
    case ZodIssueCode.custom:
      message = `Invalid input`;
      break;
    case ZodIssueCode.invalid_intersection_types:
      message = `Intersection results could not be merged`;
      break;
    case ZodIssueCode.not_multiple_of:
      message = `Number must be a multiple of ${issue.multipleOf}`;
      break;
    case ZodIssueCode.not_finite:
      message = "Number must be finite";
      break;
    default:
      message = _ctx.defaultError;
      util.assertNever(issue);
  }
  return { message };
};
var en_default = errorMap;

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/node_modules/zod/v3/errors.js
var overrideErrorMap = en_default;
function setErrorMap(map) {
  overrideErrorMap = map;
}
function getErrorMap() {
  return overrideErrorMap;
}

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/node_modules/zod/v3/helpers/parseUtil.js
var makeIssue = (params) => {
  const { data, path, errorMaps, issueData } = params;
  const fullPath = [...path, ...issueData.path || []];
  const fullIssue = {
    ...issueData,
    path: fullPath
  };
  if (issueData.message !== void 0) {
    return {
      ...issueData,
      path: fullPath,
      message: issueData.message
    };
  }
  let errorMessage = "";
  const maps = errorMaps.filter((m) => !!m).slice().reverse();
  for (const map of maps) {
    errorMessage = map(fullIssue, { data, defaultError: errorMessage }).message;
  }
  return {
    ...issueData,
    path: fullPath,
    message: errorMessage
  };
};
var EMPTY_PATH = [];
function addIssueToContext(ctx, issueData) {
  const overrideMap = getErrorMap();
  const issue = makeIssue({
    issueData,
    data: ctx.data,
    path: ctx.path,
    errorMaps: [
      ctx.common.contextualErrorMap,
      // contextual error map is first priority
      ctx.schemaErrorMap,
      // then schema-bound map if available
      overrideMap,
      // then global override map
      overrideMap === en_default ? void 0 : en_default
      // then global default map
    ].filter((x) => !!x)
  });
  ctx.common.issues.push(issue);
}
var ParseStatus = class _ParseStatus {
  constructor() {
    this.value = "valid";
  }
  dirty() {
    if (this.value === "valid")
      this.value = "dirty";
  }
  abort() {
    if (this.value !== "aborted")
      this.value = "aborted";
  }
  static mergeArray(status, results) {
    const arrayValue = [];
    for (const s of results) {
      if (s.status === "aborted")
        return INVALID;
      if (s.status === "dirty")
        status.dirty();
      arrayValue.push(s.value);
    }
    return { status: status.value, value: arrayValue };
  }
  static async mergeObjectAsync(status, pairs) {
    const syncPairs = [];
    for (const pair of pairs) {
      const key = await pair.key;
      const value = await pair.value;
      syncPairs.push({
        key,
        value
      });
    }
    return _ParseStatus.mergeObjectSync(status, syncPairs);
  }
  static mergeObjectSync(status, pairs) {
    const finalObject = {};
    for (const pair of pairs) {
      const { key, value } = pair;
      if (key.status === "aborted")
        return INVALID;
      if (value.status === "aborted")
        return INVALID;
      if (key.status === "dirty")
        status.dirty();
      if (value.status === "dirty")
        status.dirty();
      if (key.value !== "__proto__" && (typeof value.value !== "undefined" || pair.alwaysSet)) {
        finalObject[key.value] = value.value;
      }
    }
    return { status: status.value, value: finalObject };
  }
};
var INVALID = Object.freeze({
  status: "aborted"
});
var DIRTY = (value) => ({ status: "dirty", value });
var OK = (value) => ({ status: "valid", value });
var isAborted = (x) => x.status === "aborted";
var isDirty = (x) => x.status === "dirty";
var isValid = (x) => x.status === "valid";
var isAsync = (x) => typeof Promise !== "undefined" && x instanceof Promise;

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/node_modules/zod/v3/helpers/errorUtil.js
var errorUtil;
(function(errorUtil2) {
  errorUtil2.errToObj = (message) => typeof message === "string" ? { message } : message || {};
  errorUtil2.toString = (message) => typeof message === "string" ? message : message?.message;
})(errorUtil || (errorUtil = {}));

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/node_modules/zod/v3/types.js
var ParseInputLazyPath = class {
  constructor(parent, value, path, key) {
    this._cachedPath = [];
    this.parent = parent;
    this.data = value;
    this._path = path;
    this._key = key;
  }
  get path() {
    if (!this._cachedPath.length) {
      if (Array.isArray(this._key)) {
        this._cachedPath.push(...this._path, ...this._key);
      } else {
        this._cachedPath.push(...this._path, this._key);
      }
    }
    return this._cachedPath;
  }
};
var handleResult = (ctx, result) => {
  if (isValid(result)) {
    return { success: true, data: result.value };
  } else {
    if (!ctx.common.issues.length) {
      throw new Error("Validation failed but no issues detected.");
    }
    return {
      success: false,
      get error() {
        if (this._error)
          return this._error;
        const error = new ZodError(ctx.common.issues);
        this._error = error;
        return this._error;
      }
    };
  }
};
function processCreateParams(params) {
  if (!params)
    return {};
  const { errorMap: errorMap2, invalid_type_error, required_error, description } = params;
  if (errorMap2 && (invalid_type_error || required_error)) {
    throw new Error(`Can't use "invalid_type_error" or "required_error" in conjunction with custom error map.`);
  }
  if (errorMap2)
    return { errorMap: errorMap2, description };
  const customMap = (iss, ctx) => {
    const { message } = params;
    if (iss.code === "invalid_enum_value") {
      return { message: message ?? ctx.defaultError };
    }
    if (typeof ctx.data === "undefined") {
      return { message: message ?? required_error ?? ctx.defaultError };
    }
    if (iss.code !== "invalid_type")
      return { message: ctx.defaultError };
    return { message: message ?? invalid_type_error ?? ctx.defaultError };
  };
  return { errorMap: customMap, description };
}
var ZodType = class {
  get description() {
    return this._def.description;
  }
  _getType(input) {
    return getParsedType(input.data);
  }
  _getOrReturnCtx(input, ctx) {
    return ctx || {
      common: input.parent.common,
      data: input.data,
      parsedType: getParsedType(input.data),
      schemaErrorMap: this._def.errorMap,
      path: input.path,
      parent: input.parent
    };
  }
  _processInputParams(input) {
    return {
      status: new ParseStatus(),
      ctx: {
        common: input.parent.common,
        data: input.data,
        parsedType: getParsedType(input.data),
        schemaErrorMap: this._def.errorMap,
        path: input.path,
        parent: input.parent
      }
    };
  }
  _parseSync(input) {
    const result = this._parse(input);
    if (isAsync(result)) {
      throw new Error("Synchronous parse encountered promise.");
    }
    return result;
  }
  _parseAsync(input) {
    const result = this._parse(input);
    return Promise.resolve(result);
  }
  parse(data, params) {
    const result = this.safeParse(data, params);
    if (result.success)
      return result.data;
    throw result.error;
  }
  safeParse(data, params) {
    const ctx = {
      common: {
        issues: [],
        async: params?.async ?? false,
        contextualErrorMap: params?.errorMap
      },
      path: params?.path || [],
      schemaErrorMap: this._def.errorMap,
      parent: null,
      data,
      parsedType: getParsedType(data)
    };
    const result = this._parseSync({ data, path: ctx.path, parent: ctx });
    return handleResult(ctx, result);
  }
  "~validate"(data) {
    const ctx = {
      common: {
        issues: [],
        async: !!this["~standard"].async
      },
      path: [],
      schemaErrorMap: this._def.errorMap,
      parent: null,
      data,
      parsedType: getParsedType(data)
    };
    if (!this["~standard"].async) {
      try {
        const result = this._parseSync({ data, path: [], parent: ctx });
        return isValid(result) ? {
          value: result.value
        } : {
          issues: ctx.common.issues
        };
      } catch (err) {
        if (err?.message?.toLowerCase()?.includes("encountered")) {
          this["~standard"].async = true;
        }
        ctx.common = {
          issues: [],
          async: true
        };
      }
    }
    return this._parseAsync({ data, path: [], parent: ctx }).then((result) => isValid(result) ? {
      value: result.value
    } : {
      issues: ctx.common.issues
    });
  }
  async parseAsync(data, params) {
    const result = await this.safeParseAsync(data, params);
    if (result.success)
      return result.data;
    throw result.error;
  }
  async safeParseAsync(data, params) {
    const ctx = {
      common: {
        issues: [],
        contextualErrorMap: params?.errorMap,
        async: true
      },
      path: params?.path || [],
      schemaErrorMap: this._def.errorMap,
      parent: null,
      data,
      parsedType: getParsedType(data)
    };
    const maybeAsyncResult = this._parse({ data, path: ctx.path, parent: ctx });
    const result = await (isAsync(maybeAsyncResult) ? maybeAsyncResult : Promise.resolve(maybeAsyncResult));
    return handleResult(ctx, result);
  }
  refine(check, message) {
    const getIssueProperties = (val) => {
      if (typeof message === "string" || typeof message === "undefined") {
        return { message };
      } else if (typeof message === "function") {
        return message(val);
      } else {
        return message;
      }
    };
    return this._refinement((val, ctx) => {
      const result = check(val);
      const setError = () => ctx.addIssue({
        code: ZodIssueCode.custom,
        ...getIssueProperties(val)
      });
      if (typeof Promise !== "undefined" && result instanceof Promise) {
        return result.then((data) => {
          if (!data) {
            setError();
            return false;
          } else {
            return true;
          }
        });
      }
      if (!result) {
        setError();
        return false;
      } else {
        return true;
      }
    });
  }
  refinement(check, refinementData) {
    return this._refinement((val, ctx) => {
      if (!check(val)) {
        ctx.addIssue(typeof refinementData === "function" ? refinementData(val, ctx) : refinementData);
        return false;
      } else {
        return true;
      }
    });
  }
  _refinement(refinement) {
    return new ZodEffects({
      schema: this,
      typeName: ZodFirstPartyTypeKind.ZodEffects,
      effect: { type: "refinement", refinement }
    });
  }
  superRefine(refinement) {
    return this._refinement(refinement);
  }
  constructor(def) {
    this.spa = this.safeParseAsync;
    this._def = def;
    this.parse = this.parse.bind(this);
    this.safeParse = this.safeParse.bind(this);
    this.parseAsync = this.parseAsync.bind(this);
    this.safeParseAsync = this.safeParseAsync.bind(this);
    this.spa = this.spa.bind(this);
    this.refine = this.refine.bind(this);
    this.refinement = this.refinement.bind(this);
    this.superRefine = this.superRefine.bind(this);
    this.optional = this.optional.bind(this);
    this.nullable = this.nullable.bind(this);
    this.nullish = this.nullish.bind(this);
    this.array = this.array.bind(this);
    this.promise = this.promise.bind(this);
    this.or = this.or.bind(this);
    this.and = this.and.bind(this);
    this.transform = this.transform.bind(this);
    this.brand = this.brand.bind(this);
    this.default = this.default.bind(this);
    this.catch = this.catch.bind(this);
    this.describe = this.describe.bind(this);
    this.pipe = this.pipe.bind(this);
    this.readonly = this.readonly.bind(this);
    this.isNullable = this.isNullable.bind(this);
    this.isOptional = this.isOptional.bind(this);
    this["~standard"] = {
      version: 1,
      vendor: "zod",
      validate: (data) => this["~validate"](data)
    };
  }
  optional() {
    return ZodOptional.create(this, this._def);
  }
  nullable() {
    return ZodNullable.create(this, this._def);
  }
  nullish() {
    return this.nullable().optional();
  }
  array() {
    return ZodArray.create(this);
  }
  promise() {
    return ZodPromise.create(this, this._def);
  }
  or(option) {
    return ZodUnion.create([this, option], this._def);
  }
  and(incoming) {
    return ZodIntersection.create(this, incoming, this._def);
  }
  transform(transform) {
    return new ZodEffects({
      ...processCreateParams(this._def),
      schema: this,
      typeName: ZodFirstPartyTypeKind.ZodEffects,
      effect: { type: "transform", transform }
    });
  }
  default(def) {
    const defaultValueFunc = typeof def === "function" ? def : () => def;
    return new ZodDefault({
      ...processCreateParams(this._def),
      innerType: this,
      defaultValue: defaultValueFunc,
      typeName: ZodFirstPartyTypeKind.ZodDefault
    });
  }
  brand() {
    return new ZodBranded({
      typeName: ZodFirstPartyTypeKind.ZodBranded,
      type: this,
      ...processCreateParams(this._def)
    });
  }
  catch(def) {
    const catchValueFunc = typeof def === "function" ? def : () => def;
    return new ZodCatch({
      ...processCreateParams(this._def),
      innerType: this,
      catchValue: catchValueFunc,
      typeName: ZodFirstPartyTypeKind.ZodCatch
    });
  }
  describe(description) {
    const This = this.constructor;
    return new This({
      ...this._def,
      description
    });
  }
  pipe(target) {
    return ZodPipeline.create(this, target);
  }
  readonly() {
    return ZodReadonly.create(this);
  }
  isOptional() {
    return this.safeParse(void 0).success;
  }
  isNullable() {
    return this.safeParse(null).success;
  }
};
var cuidRegex = /^c[^\s-]{8,}$/i;
var cuid2Regex = /^[0-9a-z]+$/;
var ulidRegex = /^[0-9A-HJKMNP-TV-Z]{26}$/i;
var uuidRegex = /^[0-9a-fA-F]{8}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{12}$/i;
var nanoidRegex = /^[a-z0-9_-]{21}$/i;
var jwtRegex = /^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*$/;
var durationRegex = /^[-+]?P(?!$)(?:(?:[-+]?\d+Y)|(?:[-+]?\d+[.,]\d+Y$))?(?:(?:[-+]?\d+M)|(?:[-+]?\d+[.,]\d+M$))?(?:(?:[-+]?\d+W)|(?:[-+]?\d+[.,]\d+W$))?(?:(?:[-+]?\d+D)|(?:[-+]?\d+[.,]\d+D$))?(?:T(?=[\d+-])(?:(?:[-+]?\d+H)|(?:[-+]?\d+[.,]\d+H$))?(?:(?:[-+]?\d+M)|(?:[-+]?\d+[.,]\d+M$))?(?:[-+]?\d+(?:[.,]\d+)?S)?)??$/;
var emailRegex = /^(?!\.)(?!.*\.\.)([A-Z0-9_'+\-\.]*)[A-Z0-9_+-]@([A-Z0-9][A-Z0-9\-]*\.)+[A-Z]{2,}$/i;
var _emojiRegex = `^(\\p{Extended_Pictographic}|\\p{Emoji_Component})+$`;
var emojiRegex;
var ipv4Regex = /^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$/;
var ipv4CidrRegex = /^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\/(3[0-2]|[12]?[0-9])$/;
var ipv6Regex = /^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$/;
var ipv6CidrRegex = /^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))\/(12[0-8]|1[01][0-9]|[1-9]?[0-9])$/;
var base64Regex = /^([0-9a-zA-Z+/]{4})*(([0-9a-zA-Z+/]{2}==)|([0-9a-zA-Z+/]{3}=))?$/;
var base64urlRegex = /^([0-9a-zA-Z-_]{4})*(([0-9a-zA-Z-_]{2}(==)?)|([0-9a-zA-Z-_]{3}(=)?))?$/;
var dateRegexSource = `((\\d\\d[2468][048]|\\d\\d[13579][26]|\\d\\d0[48]|[02468][048]00|[13579][26]00)-02-29|\\d{4}-((0[13578]|1[02])-(0[1-9]|[12]\\d|3[01])|(0[469]|11)-(0[1-9]|[12]\\d|30)|(02)-(0[1-9]|1\\d|2[0-8])))`;
var dateRegex = new RegExp(`^${dateRegexSource}$`);
function timeRegexSource(args) {
  let secondsRegexSource = `[0-5]\\d`;
  if (args.precision) {
    secondsRegexSource = `${secondsRegexSource}\\.\\d{${args.precision}}`;
  } else if (args.precision == null) {
    secondsRegexSource = `${secondsRegexSource}(\\.\\d+)?`;
  }
  const secondsQuantifier = args.precision ? "+" : "?";
  return `([01]\\d|2[0-3]):[0-5]\\d(:${secondsRegexSource})${secondsQuantifier}`;
}
function timeRegex(args) {
  return new RegExp(`^${timeRegexSource(args)}$`);
}
function datetimeRegex(args) {
  let regex = `${dateRegexSource}T${timeRegexSource(args)}`;
  const opts = [];
  opts.push(args.local ? `Z?` : `Z`);
  if (args.offset)
    opts.push(`([+-]\\d{2}:?\\d{2})`);
  regex = `${regex}(${opts.join("|")})`;
  return new RegExp(`^${regex}$`);
}
function isValidIP(ip, version) {
  if ((version === "v4" || !version) && ipv4Regex.test(ip)) {
    return true;
  }
  if ((version === "v6" || !version) && ipv6Regex.test(ip)) {
    return true;
  }
  return false;
}
function isValidJWT(jwt, alg) {
  if (!jwtRegex.test(jwt))
    return false;
  try {
    const [header] = jwt.split(".");
    if (!header)
      return false;
    const base64 = header.replace(/-/g, "+").replace(/_/g, "/").padEnd(header.length + (4 - header.length % 4) % 4, "=");
    const decoded = JSON.parse(atob(base64));
    if (typeof decoded !== "object" || decoded === null)
      return false;
    if ("typ" in decoded && decoded?.typ !== "JWT")
      return false;
    if (!decoded.alg)
      return false;
    if (alg && decoded.alg !== alg)
      return false;
    return true;
  } catch {
    return false;
  }
}
function isValidCidr(ip, version) {
  if ((version === "v4" || !version) && ipv4CidrRegex.test(ip)) {
    return true;
  }
  if ((version === "v6" || !version) && ipv6CidrRegex.test(ip)) {
    return true;
  }
  return false;
}
var ZodString = class _ZodString extends ZodType {
  _parse(input) {
    if (this._def.coerce) {
      input.data = String(input.data);
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.string) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.string,
        received: ctx2.parsedType
      });
      return INVALID;
    }
    const status = new ParseStatus();
    let ctx = void 0;
    for (const check of this._def.checks) {
      if (check.kind === "min") {
        if (input.data.length < check.value) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_small,
            minimum: check.value,
            type: "string",
            inclusive: true,
            exact: false,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "max") {
        if (input.data.length > check.value) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_big,
            maximum: check.value,
            type: "string",
            inclusive: true,
            exact: false,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "length") {
        const tooBig = input.data.length > check.value;
        const tooSmall = input.data.length < check.value;
        if (tooBig || tooSmall) {
          ctx = this._getOrReturnCtx(input, ctx);
          if (tooBig) {
            addIssueToContext(ctx, {
              code: ZodIssueCode.too_big,
              maximum: check.value,
              type: "string",
              inclusive: true,
              exact: true,
              message: check.message
            });
          } else if (tooSmall) {
            addIssueToContext(ctx, {
              code: ZodIssueCode.too_small,
              minimum: check.value,
              type: "string",
              inclusive: true,
              exact: true,
              message: check.message
            });
          }
          status.dirty();
        }
      } else if (check.kind === "email") {
        if (!emailRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "email",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "emoji") {
        if (!emojiRegex) {
          emojiRegex = new RegExp(_emojiRegex, "u");
        }
        if (!emojiRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "emoji",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "uuid") {
        if (!uuidRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "uuid",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "nanoid") {
        if (!nanoidRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "nanoid",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "cuid") {
        if (!cuidRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "cuid",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "cuid2") {
        if (!cuid2Regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "cuid2",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "ulid") {
        if (!ulidRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "ulid",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "url") {
        try {
          new URL(input.data);
        } catch {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "url",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "regex") {
        check.regex.lastIndex = 0;
        const testResult = check.regex.test(input.data);
        if (!testResult) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "regex",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "trim") {
        input.data = input.data.trim();
      } else if (check.kind === "includes") {
        if (!input.data.includes(check.value, check.position)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: { includes: check.value, position: check.position },
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "toLowerCase") {
        input.data = input.data.toLowerCase();
      } else if (check.kind === "toUpperCase") {
        input.data = input.data.toUpperCase();
      } else if (check.kind === "startsWith") {
        if (!input.data.startsWith(check.value)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: { startsWith: check.value },
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "endsWith") {
        if (!input.data.endsWith(check.value)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: { endsWith: check.value },
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "datetime") {
        const regex = datetimeRegex(check);
        if (!regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: "datetime",
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "date") {
        const regex = dateRegex;
        if (!regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: "date",
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "time") {
        const regex = timeRegex(check);
        if (!regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: "time",
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "duration") {
        if (!durationRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "duration",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "ip") {
        if (!isValidIP(input.data, check.version)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "ip",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "jwt") {
        if (!isValidJWT(input.data, check.alg)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "jwt",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "cidr") {
        if (!isValidCidr(input.data, check.version)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "cidr",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "base64") {
        if (!base64Regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "base64",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "base64url") {
        if (!base64urlRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "base64url",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else {
        util.assertNever(check);
      }
    }
    return { status: status.value, value: input.data };
  }
  _regex(regex, validation, message) {
    return this.refinement((data) => regex.test(data), {
      validation,
      code: ZodIssueCode.invalid_string,
      ...errorUtil.errToObj(message)
    });
  }
  _addCheck(check) {
    return new _ZodString({
      ...this._def,
      checks: [...this._def.checks, check]
    });
  }
  email(message) {
    return this._addCheck({ kind: "email", ...errorUtil.errToObj(message) });
  }
  url(message) {
    return this._addCheck({ kind: "url", ...errorUtil.errToObj(message) });
  }
  emoji(message) {
    return this._addCheck({ kind: "emoji", ...errorUtil.errToObj(message) });
  }
  uuid(message) {
    return this._addCheck({ kind: "uuid", ...errorUtil.errToObj(message) });
  }
  nanoid(message) {
    return this._addCheck({ kind: "nanoid", ...errorUtil.errToObj(message) });
  }
  cuid(message) {
    return this._addCheck({ kind: "cuid", ...errorUtil.errToObj(message) });
  }
  cuid2(message) {
    return this._addCheck({ kind: "cuid2", ...errorUtil.errToObj(message) });
  }
  ulid(message) {
    return this._addCheck({ kind: "ulid", ...errorUtil.errToObj(message) });
  }
  base64(message) {
    return this._addCheck({ kind: "base64", ...errorUtil.errToObj(message) });
  }
  base64url(message) {
    return this._addCheck({
      kind: "base64url",
      ...errorUtil.errToObj(message)
    });
  }
  jwt(options) {
    return this._addCheck({ kind: "jwt", ...errorUtil.errToObj(options) });
  }
  ip(options) {
    return this._addCheck({ kind: "ip", ...errorUtil.errToObj(options) });
  }
  cidr(options) {
    return this._addCheck({ kind: "cidr", ...errorUtil.errToObj(options) });
  }
  datetime(options) {
    if (typeof options === "string") {
      return this._addCheck({
        kind: "datetime",
        precision: null,
        offset: false,
        local: false,
        message: options
      });
    }
    return this._addCheck({
      kind: "datetime",
      precision: typeof options?.precision === "undefined" ? null : options?.precision,
      offset: options?.offset ?? false,
      local: options?.local ?? false,
      ...errorUtil.errToObj(options?.message)
    });
  }
  date(message) {
    return this._addCheck({ kind: "date", message });
  }
  time(options) {
    if (typeof options === "string") {
      return this._addCheck({
        kind: "time",
        precision: null,
        message: options
      });
    }
    return this._addCheck({
      kind: "time",
      precision: typeof options?.precision === "undefined" ? null : options?.precision,
      ...errorUtil.errToObj(options?.message)
    });
  }
  duration(message) {
    return this._addCheck({ kind: "duration", ...errorUtil.errToObj(message) });
  }
  regex(regex, message) {
    return this._addCheck({
      kind: "regex",
      regex,
      ...errorUtil.errToObj(message)
    });
  }
  includes(value, options) {
    return this._addCheck({
      kind: "includes",
      value,
      position: options?.position,
      ...errorUtil.errToObj(options?.message)
    });
  }
  startsWith(value, message) {
    return this._addCheck({
      kind: "startsWith",
      value,
      ...errorUtil.errToObj(message)
    });
  }
  endsWith(value, message) {
    return this._addCheck({
      kind: "endsWith",
      value,
      ...errorUtil.errToObj(message)
    });
  }
  min(minLength, message) {
    return this._addCheck({
      kind: "min",
      value: minLength,
      ...errorUtil.errToObj(message)
    });
  }
  max(maxLength, message) {
    return this._addCheck({
      kind: "max",
      value: maxLength,
      ...errorUtil.errToObj(message)
    });
  }
  length(len, message) {
    return this._addCheck({
      kind: "length",
      value: len,
      ...errorUtil.errToObj(message)
    });
  }
  /**
   * Equivalent to `.min(1)`
   */
  nonempty(message) {
    return this.min(1, errorUtil.errToObj(message));
  }
  trim() {
    return new _ZodString({
      ...this._def,
      checks: [...this._def.checks, { kind: "trim" }]
    });
  }
  toLowerCase() {
    return new _ZodString({
      ...this._def,
      checks: [...this._def.checks, { kind: "toLowerCase" }]
    });
  }
  toUpperCase() {
    return new _ZodString({
      ...this._def,
      checks: [...this._def.checks, { kind: "toUpperCase" }]
    });
  }
  get isDatetime() {
    return !!this._def.checks.find((ch) => ch.kind === "datetime");
  }
  get isDate() {
    return !!this._def.checks.find((ch) => ch.kind === "date");
  }
  get isTime() {
    return !!this._def.checks.find((ch) => ch.kind === "time");
  }
  get isDuration() {
    return !!this._def.checks.find((ch) => ch.kind === "duration");
  }
  get isEmail() {
    return !!this._def.checks.find((ch) => ch.kind === "email");
  }
  get isURL() {
    return !!this._def.checks.find((ch) => ch.kind === "url");
  }
  get isEmoji() {
    return !!this._def.checks.find((ch) => ch.kind === "emoji");
  }
  get isUUID() {
    return !!this._def.checks.find((ch) => ch.kind === "uuid");
  }
  get isNANOID() {
    return !!this._def.checks.find((ch) => ch.kind === "nanoid");
  }
  get isCUID() {
    return !!this._def.checks.find((ch) => ch.kind === "cuid");
  }
  get isCUID2() {
    return !!this._def.checks.find((ch) => ch.kind === "cuid2");
  }
  get isULID() {
    return !!this._def.checks.find((ch) => ch.kind === "ulid");
  }
  get isIP() {
    return !!this._def.checks.find((ch) => ch.kind === "ip");
  }
  get isCIDR() {
    return !!this._def.checks.find((ch) => ch.kind === "cidr");
  }
  get isBase64() {
    return !!this._def.checks.find((ch) => ch.kind === "base64");
  }
  get isBase64url() {
    return !!this._def.checks.find((ch) => ch.kind === "base64url");
  }
  get minLength() {
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      }
    }
    return min;
  }
  get maxLength() {
    let max = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return max;
  }
};
ZodString.create = (params) => {
  return new ZodString({
    checks: [],
    typeName: ZodFirstPartyTypeKind.ZodString,
    coerce: params?.coerce ?? false,
    ...processCreateParams(params)
  });
};
function floatSafeRemainder(val, step) {
  const valDecCount = (val.toString().split(".")[1] || "").length;
  const stepDecCount = (step.toString().split(".")[1] || "").length;
  const decCount = valDecCount > stepDecCount ? valDecCount : stepDecCount;
  const valInt = Number.parseInt(val.toFixed(decCount).replace(".", ""));
  const stepInt = Number.parseInt(step.toFixed(decCount).replace(".", ""));
  return valInt % stepInt / 10 ** decCount;
}
var ZodNumber = class _ZodNumber extends ZodType {
  constructor() {
    super(...arguments);
    this.min = this.gte;
    this.max = this.lte;
    this.step = this.multipleOf;
  }
  _parse(input) {
    if (this._def.coerce) {
      input.data = Number(input.data);
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.number) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.number,
        received: ctx2.parsedType
      });
      return INVALID;
    }
    let ctx = void 0;
    const status = new ParseStatus();
    for (const check of this._def.checks) {
      if (check.kind === "int") {
        if (!util.isInteger(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_type,
            expected: "integer",
            received: "float",
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "min") {
        const tooSmall = check.inclusive ? input.data < check.value : input.data <= check.value;
        if (tooSmall) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_small,
            minimum: check.value,
            type: "number",
            inclusive: check.inclusive,
            exact: false,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "max") {
        const tooBig = check.inclusive ? input.data > check.value : input.data >= check.value;
        if (tooBig) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_big,
            maximum: check.value,
            type: "number",
            inclusive: check.inclusive,
            exact: false,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "multipleOf") {
        if (floatSafeRemainder(input.data, check.value) !== 0) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.not_multiple_of,
            multipleOf: check.value,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "finite") {
        if (!Number.isFinite(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.not_finite,
            message: check.message
          });
          status.dirty();
        }
      } else {
        util.assertNever(check);
      }
    }
    return { status: status.value, value: input.data };
  }
  gte(value, message) {
    return this.setLimit("min", value, true, errorUtil.toString(message));
  }
  gt(value, message) {
    return this.setLimit("min", value, false, errorUtil.toString(message));
  }
  lte(value, message) {
    return this.setLimit("max", value, true, errorUtil.toString(message));
  }
  lt(value, message) {
    return this.setLimit("max", value, false, errorUtil.toString(message));
  }
  setLimit(kind, value, inclusive, message) {
    return new _ZodNumber({
      ...this._def,
      checks: [
        ...this._def.checks,
        {
          kind,
          value,
          inclusive,
          message: errorUtil.toString(message)
        }
      ]
    });
  }
  _addCheck(check) {
    return new _ZodNumber({
      ...this._def,
      checks: [...this._def.checks, check]
    });
  }
  int(message) {
    return this._addCheck({
      kind: "int",
      message: errorUtil.toString(message)
    });
  }
  positive(message) {
    return this._addCheck({
      kind: "min",
      value: 0,
      inclusive: false,
      message: errorUtil.toString(message)
    });
  }
  negative(message) {
    return this._addCheck({
      kind: "max",
      value: 0,
      inclusive: false,
      message: errorUtil.toString(message)
    });
  }
  nonpositive(message) {
    return this._addCheck({
      kind: "max",
      value: 0,
      inclusive: true,
      message: errorUtil.toString(message)
    });
  }
  nonnegative(message) {
    return this._addCheck({
      kind: "min",
      value: 0,
      inclusive: true,
      message: errorUtil.toString(message)
    });
  }
  multipleOf(value, message) {
    return this._addCheck({
      kind: "multipleOf",
      value,
      message: errorUtil.toString(message)
    });
  }
  finite(message) {
    return this._addCheck({
      kind: "finite",
      message: errorUtil.toString(message)
    });
  }
  safe(message) {
    return this._addCheck({
      kind: "min",
      inclusive: true,
      value: Number.MIN_SAFE_INTEGER,
      message: errorUtil.toString(message)
    })._addCheck({
      kind: "max",
      inclusive: true,
      value: Number.MAX_SAFE_INTEGER,
      message: errorUtil.toString(message)
    });
  }
  get minValue() {
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      }
    }
    return min;
  }
  get maxValue() {
    let max = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return max;
  }
  get isInt() {
    return !!this._def.checks.find((ch) => ch.kind === "int" || ch.kind === "multipleOf" && util.isInteger(ch.value));
  }
  get isFinite() {
    let max = null;
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "finite" || ch.kind === "int" || ch.kind === "multipleOf") {
        return true;
      } else if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      } else if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return Number.isFinite(min) && Number.isFinite(max);
  }
};
ZodNumber.create = (params) => {
  return new ZodNumber({
    checks: [],
    typeName: ZodFirstPartyTypeKind.ZodNumber,
    coerce: params?.coerce || false,
    ...processCreateParams(params)
  });
};
var ZodBigInt = class _ZodBigInt extends ZodType {
  constructor() {
    super(...arguments);
    this.min = this.gte;
    this.max = this.lte;
  }
  _parse(input) {
    if (this._def.coerce) {
      try {
        input.data = BigInt(input.data);
      } catch {
        return this._getInvalidInput(input);
      }
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.bigint) {
      return this._getInvalidInput(input);
    }
    let ctx = void 0;
    const status = new ParseStatus();
    for (const check of this._def.checks) {
      if (check.kind === "min") {
        const tooSmall = check.inclusive ? input.data < check.value : input.data <= check.value;
        if (tooSmall) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_small,
            type: "bigint",
            minimum: check.value,
            inclusive: check.inclusive,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "max") {
        const tooBig = check.inclusive ? input.data > check.value : input.data >= check.value;
        if (tooBig) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_big,
            type: "bigint",
            maximum: check.value,
            inclusive: check.inclusive,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "multipleOf") {
        if (input.data % check.value !== BigInt(0)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.not_multiple_of,
            multipleOf: check.value,
            message: check.message
          });
          status.dirty();
        }
      } else {
        util.assertNever(check);
      }
    }
    return { status: status.value, value: input.data };
  }
  _getInvalidInput(input) {
    const ctx = this._getOrReturnCtx(input);
    addIssueToContext(ctx, {
      code: ZodIssueCode.invalid_type,
      expected: ZodParsedType.bigint,
      received: ctx.parsedType
    });
    return INVALID;
  }
  gte(value, message) {
    return this.setLimit("min", value, true, errorUtil.toString(message));
  }
  gt(value, message) {
    return this.setLimit("min", value, false, errorUtil.toString(message));
  }
  lte(value, message) {
    return this.setLimit("max", value, true, errorUtil.toString(message));
  }
  lt(value, message) {
    return this.setLimit("max", value, false, errorUtil.toString(message));
  }
  setLimit(kind, value, inclusive, message) {
    return new _ZodBigInt({
      ...this._def,
      checks: [
        ...this._def.checks,
        {
          kind,
          value,
          inclusive,
          message: errorUtil.toString(message)
        }
      ]
    });
  }
  _addCheck(check) {
    return new _ZodBigInt({
      ...this._def,
      checks: [...this._def.checks, check]
    });
  }
  positive(message) {
    return this._addCheck({
      kind: "min",
      value: BigInt(0),
      inclusive: false,
      message: errorUtil.toString(message)
    });
  }
  negative(message) {
    return this._addCheck({
      kind: "max",
      value: BigInt(0),
      inclusive: false,
      message: errorUtil.toString(message)
    });
  }
  nonpositive(message) {
    return this._addCheck({
      kind: "max",
      value: BigInt(0),
      inclusive: true,
      message: errorUtil.toString(message)
    });
  }
  nonnegative(message) {
    return this._addCheck({
      kind: "min",
      value: BigInt(0),
      inclusive: true,
      message: errorUtil.toString(message)
    });
  }
  multipleOf(value, message) {
    return this._addCheck({
      kind: "multipleOf",
      value,
      message: errorUtil.toString(message)
    });
  }
  get minValue() {
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      }
    }
    return min;
  }
  get maxValue() {
    let max = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return max;
  }
};
ZodBigInt.create = (params) => {
  return new ZodBigInt({
    checks: [],
    typeName: ZodFirstPartyTypeKind.ZodBigInt,
    coerce: params?.coerce ?? false,
    ...processCreateParams(params)
  });
};
var ZodBoolean = class extends ZodType {
  _parse(input) {
    if (this._def.coerce) {
      input.data = Boolean(input.data);
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.boolean) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.boolean,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
};
ZodBoolean.create = (params) => {
  return new ZodBoolean({
    typeName: ZodFirstPartyTypeKind.ZodBoolean,
    coerce: params?.coerce || false,
    ...processCreateParams(params)
  });
};
var ZodDate = class _ZodDate extends ZodType {
  _parse(input) {
    if (this._def.coerce) {
      input.data = new Date(input.data);
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.date) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.date,
        received: ctx2.parsedType
      });
      return INVALID;
    }
    if (Number.isNaN(input.data.getTime())) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_date
      });
      return INVALID;
    }
    const status = new ParseStatus();
    let ctx = void 0;
    for (const check of this._def.checks) {
      if (check.kind === "min") {
        if (input.data.getTime() < check.value) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_small,
            message: check.message,
            inclusive: true,
            exact: false,
            minimum: check.value,
            type: "date"
          });
          status.dirty();
        }
      } else if (check.kind === "max") {
        if (input.data.getTime() > check.value) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_big,
            message: check.message,
            inclusive: true,
            exact: false,
            maximum: check.value,
            type: "date"
          });
          status.dirty();
        }
      } else {
        util.assertNever(check);
      }
    }
    return {
      status: status.value,
      value: new Date(input.data.getTime())
    };
  }
  _addCheck(check) {
    return new _ZodDate({
      ...this._def,
      checks: [...this._def.checks, check]
    });
  }
  min(minDate, message) {
    return this._addCheck({
      kind: "min",
      value: minDate.getTime(),
      message: errorUtil.toString(message)
    });
  }
  max(maxDate, message) {
    return this._addCheck({
      kind: "max",
      value: maxDate.getTime(),
      message: errorUtil.toString(message)
    });
  }
  get minDate() {
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      }
    }
    return min != null ? new Date(min) : null;
  }
  get maxDate() {
    let max = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return max != null ? new Date(max) : null;
  }
};
ZodDate.create = (params) => {
  return new ZodDate({
    checks: [],
    coerce: params?.coerce || false,
    typeName: ZodFirstPartyTypeKind.ZodDate,
    ...processCreateParams(params)
  });
};
var ZodSymbol = class extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.symbol) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.symbol,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
};
ZodSymbol.create = (params) => {
  return new ZodSymbol({
    typeName: ZodFirstPartyTypeKind.ZodSymbol,
    ...processCreateParams(params)
  });
};
var ZodUndefined = class extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.undefined) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.undefined,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
};
ZodUndefined.create = (params) => {
  return new ZodUndefined({
    typeName: ZodFirstPartyTypeKind.ZodUndefined,
    ...processCreateParams(params)
  });
};
var ZodNull = class extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.null) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.null,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
};
ZodNull.create = (params) => {
  return new ZodNull({
    typeName: ZodFirstPartyTypeKind.ZodNull,
    ...processCreateParams(params)
  });
};
var ZodAny = class extends ZodType {
  constructor() {
    super(...arguments);
    this._any = true;
  }
  _parse(input) {
    return OK(input.data);
  }
};
ZodAny.create = (params) => {
  return new ZodAny({
    typeName: ZodFirstPartyTypeKind.ZodAny,
    ...processCreateParams(params)
  });
};
var ZodUnknown = class extends ZodType {
  constructor() {
    super(...arguments);
    this._unknown = true;
  }
  _parse(input) {
    return OK(input.data);
  }
};
ZodUnknown.create = (params) => {
  return new ZodUnknown({
    typeName: ZodFirstPartyTypeKind.ZodUnknown,
    ...processCreateParams(params)
  });
};
var ZodNever = class extends ZodType {
  _parse(input) {
    const ctx = this._getOrReturnCtx(input);
    addIssueToContext(ctx, {
      code: ZodIssueCode.invalid_type,
      expected: ZodParsedType.never,
      received: ctx.parsedType
    });
    return INVALID;
  }
};
ZodNever.create = (params) => {
  return new ZodNever({
    typeName: ZodFirstPartyTypeKind.ZodNever,
    ...processCreateParams(params)
  });
};
var ZodVoid = class extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.undefined) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.void,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
};
ZodVoid.create = (params) => {
  return new ZodVoid({
    typeName: ZodFirstPartyTypeKind.ZodVoid,
    ...processCreateParams(params)
  });
};
var ZodArray = class _ZodArray extends ZodType {
  _parse(input) {
    const { ctx, status } = this._processInputParams(input);
    const def = this._def;
    if (ctx.parsedType !== ZodParsedType.array) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.array,
        received: ctx.parsedType
      });
      return INVALID;
    }
    if (def.exactLength !== null) {
      const tooBig = ctx.data.length > def.exactLength.value;
      const tooSmall = ctx.data.length < def.exactLength.value;
      if (tooBig || tooSmall) {
        addIssueToContext(ctx, {
          code: tooBig ? ZodIssueCode.too_big : ZodIssueCode.too_small,
          minimum: tooSmall ? def.exactLength.value : void 0,
          maximum: tooBig ? def.exactLength.value : void 0,
          type: "array",
          inclusive: true,
          exact: true,
          message: def.exactLength.message
        });
        status.dirty();
      }
    }
    if (def.minLength !== null) {
      if (ctx.data.length < def.minLength.value) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.too_small,
          minimum: def.minLength.value,
          type: "array",
          inclusive: true,
          exact: false,
          message: def.minLength.message
        });
        status.dirty();
      }
    }
    if (def.maxLength !== null) {
      if (ctx.data.length > def.maxLength.value) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.too_big,
          maximum: def.maxLength.value,
          type: "array",
          inclusive: true,
          exact: false,
          message: def.maxLength.message
        });
        status.dirty();
      }
    }
    if (ctx.common.async) {
      return Promise.all([...ctx.data].map((item, i) => {
        return def.type._parseAsync(new ParseInputLazyPath(ctx, item, ctx.path, i));
      })).then((result2) => {
        return ParseStatus.mergeArray(status, result2);
      });
    }
    const result = [...ctx.data].map((item, i) => {
      return def.type._parseSync(new ParseInputLazyPath(ctx, item, ctx.path, i));
    });
    return ParseStatus.mergeArray(status, result);
  }
  get element() {
    return this._def.type;
  }
  min(minLength, message) {
    return new _ZodArray({
      ...this._def,
      minLength: { value: minLength, message: errorUtil.toString(message) }
    });
  }
  max(maxLength, message) {
    return new _ZodArray({
      ...this._def,
      maxLength: { value: maxLength, message: errorUtil.toString(message) }
    });
  }
  length(len, message) {
    return new _ZodArray({
      ...this._def,
      exactLength: { value: len, message: errorUtil.toString(message) }
    });
  }
  nonempty(message) {
    return this.min(1, message);
  }
};
ZodArray.create = (schema, params) => {
  return new ZodArray({
    type: schema,
    minLength: null,
    maxLength: null,
    exactLength: null,
    typeName: ZodFirstPartyTypeKind.ZodArray,
    ...processCreateParams(params)
  });
};
function deepPartialify(schema) {
  if (schema instanceof ZodObject) {
    const newShape = {};
    for (const key in schema.shape) {
      const fieldSchema = schema.shape[key];
      newShape[key] = ZodOptional.create(deepPartialify(fieldSchema));
    }
    return new ZodObject({
      ...schema._def,
      shape: () => newShape
    });
  } else if (schema instanceof ZodArray) {
    return new ZodArray({
      ...schema._def,
      type: deepPartialify(schema.element)
    });
  } else if (schema instanceof ZodOptional) {
    return ZodOptional.create(deepPartialify(schema.unwrap()));
  } else if (schema instanceof ZodNullable) {
    return ZodNullable.create(deepPartialify(schema.unwrap()));
  } else if (schema instanceof ZodTuple) {
    return ZodTuple.create(schema.items.map((item) => deepPartialify(item)));
  } else {
    return schema;
  }
}
var ZodObject = class _ZodObject extends ZodType {
  constructor() {
    super(...arguments);
    this._cached = null;
    this.nonstrict = this.passthrough;
    this.augment = this.extend;
  }
  _getCached() {
    if (this._cached !== null)
      return this._cached;
    const shape = this._def.shape();
    const keys = util.objectKeys(shape);
    this._cached = { shape, keys };
    return this._cached;
  }
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.object) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.object,
        received: ctx2.parsedType
      });
      return INVALID;
    }
    const { status, ctx } = this._processInputParams(input);
    const { shape, keys: shapeKeys } = this._getCached();
    const extraKeys = [];
    if (!(this._def.catchall instanceof ZodNever && this._def.unknownKeys === "strip")) {
      for (const key in ctx.data) {
        if (!shapeKeys.includes(key)) {
          extraKeys.push(key);
        }
      }
    }
    const pairs = [];
    for (const key of shapeKeys) {
      const keyValidator = shape[key];
      const value = ctx.data[key];
      pairs.push({
        key: { status: "valid", value: key },
        value: keyValidator._parse(new ParseInputLazyPath(ctx, value, ctx.path, key)),
        alwaysSet: key in ctx.data
      });
    }
    if (this._def.catchall instanceof ZodNever) {
      const unknownKeys = this._def.unknownKeys;
      if (unknownKeys === "passthrough") {
        for (const key of extraKeys) {
          pairs.push({
            key: { status: "valid", value: key },
            value: { status: "valid", value: ctx.data[key] }
          });
        }
      } else if (unknownKeys === "strict") {
        if (extraKeys.length > 0) {
          addIssueToContext(ctx, {
            code: ZodIssueCode.unrecognized_keys,
            keys: extraKeys
          });
          status.dirty();
        }
      } else if (unknownKeys === "strip") {
      } else {
        throw new Error(`Internal ZodObject error: invalid unknownKeys value.`);
      }
    } else {
      const catchall = this._def.catchall;
      for (const key of extraKeys) {
        const value = ctx.data[key];
        pairs.push({
          key: { status: "valid", value: key },
          value: catchall._parse(
            new ParseInputLazyPath(ctx, value, ctx.path, key)
            //, ctx.child(key), value, getParsedType(value)
          ),
          alwaysSet: key in ctx.data
        });
      }
    }
    if (ctx.common.async) {
      return Promise.resolve().then(async () => {
        const syncPairs = [];
        for (const pair of pairs) {
          const key = await pair.key;
          const value = await pair.value;
          syncPairs.push({
            key,
            value,
            alwaysSet: pair.alwaysSet
          });
        }
        return syncPairs;
      }).then((syncPairs) => {
        return ParseStatus.mergeObjectSync(status, syncPairs);
      });
    } else {
      return ParseStatus.mergeObjectSync(status, pairs);
    }
  }
  get shape() {
    return this._def.shape();
  }
  strict(message) {
    errorUtil.errToObj;
    return new _ZodObject({
      ...this._def,
      unknownKeys: "strict",
      ...message !== void 0 ? {
        errorMap: (issue, ctx) => {
          const defaultError = this._def.errorMap?.(issue, ctx).message ?? ctx.defaultError;
          if (issue.code === "unrecognized_keys")
            return {
              message: errorUtil.errToObj(message).message ?? defaultError
            };
          return {
            message: defaultError
          };
        }
      } : {}
    });
  }
  strip() {
    return new _ZodObject({
      ...this._def,
      unknownKeys: "strip"
    });
  }
  passthrough() {
    return new _ZodObject({
      ...this._def,
      unknownKeys: "passthrough"
    });
  }
  // const AugmentFactory =
  //   <Def extends ZodObjectDef>(def: Def) =>
  //   <Augmentation extends ZodRawShape>(
  //     augmentation: Augmentation
  //   ): ZodObject<
  //     extendShape<ReturnType<Def["shape"]>, Augmentation>,
  //     Def["unknownKeys"],
  //     Def["catchall"]
  //   > => {
  //     return new ZodObject({
  //       ...def,
  //       shape: () => ({
  //         ...def.shape(),
  //         ...augmentation,
  //       }),
  //     }) as any;
  //   };
  extend(augmentation) {
    return new _ZodObject({
      ...this._def,
      shape: () => ({
        ...this._def.shape(),
        ...augmentation
      })
    });
  }
  /**
   * Prior to zod@1.0.12 there was a bug in the
   * inferred type of merged objects. Please
   * upgrade if you are experiencing issues.
   */
  merge(merging) {
    const merged = new _ZodObject({
      unknownKeys: merging._def.unknownKeys,
      catchall: merging._def.catchall,
      shape: () => ({
        ...this._def.shape(),
        ...merging._def.shape()
      }),
      typeName: ZodFirstPartyTypeKind.ZodObject
    });
    return merged;
  }
  // merge<
  //   Incoming extends AnyZodObject,
  //   Augmentation extends Incoming["shape"],
  //   NewOutput extends {
  //     [k in keyof Augmentation | keyof Output]: k extends keyof Augmentation
  //       ? Augmentation[k]["_output"]
  //       : k extends keyof Output
  //       ? Output[k]
  //       : never;
  //   },
  //   NewInput extends {
  //     [k in keyof Augmentation | keyof Input]: k extends keyof Augmentation
  //       ? Augmentation[k]["_input"]
  //       : k extends keyof Input
  //       ? Input[k]
  //       : never;
  //   }
  // >(
  //   merging: Incoming
  // ): ZodObject<
  //   extendShape<T, ReturnType<Incoming["_def"]["shape"]>>,
  //   Incoming["_def"]["unknownKeys"],
  //   Incoming["_def"]["catchall"],
  //   NewOutput,
  //   NewInput
  // > {
  //   const merged: any = new ZodObject({
  //     unknownKeys: merging._def.unknownKeys,
  //     catchall: merging._def.catchall,
  //     shape: () =>
  //       objectUtil.mergeShapes(this._def.shape(), merging._def.shape()),
  //     typeName: ZodFirstPartyTypeKind.ZodObject,
  //   }) as any;
  //   return merged;
  // }
  setKey(key, schema) {
    return this.augment({ [key]: schema });
  }
  // merge<Incoming extends AnyZodObject>(
  //   merging: Incoming
  // ): //ZodObject<T & Incoming["_shape"], UnknownKeys, Catchall> = (merging) => {
  // ZodObject<
  //   extendShape<T, ReturnType<Incoming["_def"]["shape"]>>,
  //   Incoming["_def"]["unknownKeys"],
  //   Incoming["_def"]["catchall"]
  // > {
  //   // const mergedShape = objectUtil.mergeShapes(
  //   //   this._def.shape(),
  //   //   merging._def.shape()
  //   // );
  //   const merged: any = new ZodObject({
  //     unknownKeys: merging._def.unknownKeys,
  //     catchall: merging._def.catchall,
  //     shape: () =>
  //       objectUtil.mergeShapes(this._def.shape(), merging._def.shape()),
  //     typeName: ZodFirstPartyTypeKind.ZodObject,
  //   }) as any;
  //   return merged;
  // }
  catchall(index) {
    return new _ZodObject({
      ...this._def,
      catchall: index
    });
  }
  pick(mask) {
    const shape = {};
    for (const key of util.objectKeys(mask)) {
      if (mask[key] && this.shape[key]) {
        shape[key] = this.shape[key];
      }
    }
    return new _ZodObject({
      ...this._def,
      shape: () => shape
    });
  }
  omit(mask) {
    const shape = {};
    for (const key of util.objectKeys(this.shape)) {
      if (!mask[key]) {
        shape[key] = this.shape[key];
      }
    }
    return new _ZodObject({
      ...this._def,
      shape: () => shape
    });
  }
  /**
   * @deprecated
   */
  deepPartial() {
    return deepPartialify(this);
  }
  partial(mask) {
    const newShape = {};
    for (const key of util.objectKeys(this.shape)) {
      const fieldSchema = this.shape[key];
      if (mask && !mask[key]) {
        newShape[key] = fieldSchema;
      } else {
        newShape[key] = fieldSchema.optional();
      }
    }
    return new _ZodObject({
      ...this._def,
      shape: () => newShape
    });
  }
  required(mask) {
    const newShape = {};
    for (const key of util.objectKeys(this.shape)) {
      if (mask && !mask[key]) {
        newShape[key] = this.shape[key];
      } else {
        const fieldSchema = this.shape[key];
        let newField = fieldSchema;
        while (newField instanceof ZodOptional) {
          newField = newField._def.innerType;
        }
        newShape[key] = newField;
      }
    }
    return new _ZodObject({
      ...this._def,
      shape: () => newShape
    });
  }
  keyof() {
    return createZodEnum(util.objectKeys(this.shape));
  }
};
ZodObject.create = (shape, params) => {
  return new ZodObject({
    shape: () => shape,
    unknownKeys: "strip",
    catchall: ZodNever.create(),
    typeName: ZodFirstPartyTypeKind.ZodObject,
    ...processCreateParams(params)
  });
};
ZodObject.strictCreate = (shape, params) => {
  return new ZodObject({
    shape: () => shape,
    unknownKeys: "strict",
    catchall: ZodNever.create(),
    typeName: ZodFirstPartyTypeKind.ZodObject,
    ...processCreateParams(params)
  });
};
ZodObject.lazycreate = (shape, params) => {
  return new ZodObject({
    shape,
    unknownKeys: "strip",
    catchall: ZodNever.create(),
    typeName: ZodFirstPartyTypeKind.ZodObject,
    ...processCreateParams(params)
  });
};
var ZodUnion = class extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    const options = this._def.options;
    function handleResults(results) {
      for (const result of results) {
        if (result.result.status === "valid") {
          return result.result;
        }
      }
      for (const result of results) {
        if (result.result.status === "dirty") {
          ctx.common.issues.push(...result.ctx.common.issues);
          return result.result;
        }
      }
      const unionErrors = results.map((result) => new ZodError(result.ctx.common.issues));
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_union,
        unionErrors
      });
      return INVALID;
    }
    if (ctx.common.async) {
      return Promise.all(options.map(async (option) => {
        const childCtx = {
          ...ctx,
          common: {
            ...ctx.common,
            issues: []
          },
          parent: null
        };
        return {
          result: await option._parseAsync({
            data: ctx.data,
            path: ctx.path,
            parent: childCtx
          }),
          ctx: childCtx
        };
      })).then(handleResults);
    } else {
      let dirty = void 0;
      const issues = [];
      for (const option of options) {
        const childCtx = {
          ...ctx,
          common: {
            ...ctx.common,
            issues: []
          },
          parent: null
        };
        const result = option._parseSync({
          data: ctx.data,
          path: ctx.path,
          parent: childCtx
        });
        if (result.status === "valid") {
          return result;
        } else if (result.status === "dirty" && !dirty) {
          dirty = { result, ctx: childCtx };
        }
        if (childCtx.common.issues.length) {
          issues.push(childCtx.common.issues);
        }
      }
      if (dirty) {
        ctx.common.issues.push(...dirty.ctx.common.issues);
        return dirty.result;
      }
      const unionErrors = issues.map((issues2) => new ZodError(issues2));
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_union,
        unionErrors
      });
      return INVALID;
    }
  }
  get options() {
    return this._def.options;
  }
};
ZodUnion.create = (types, params) => {
  return new ZodUnion({
    options: types,
    typeName: ZodFirstPartyTypeKind.ZodUnion,
    ...processCreateParams(params)
  });
};
var getDiscriminator = (type) => {
  if (type instanceof ZodLazy) {
    return getDiscriminator(type.schema);
  } else if (type instanceof ZodEffects) {
    return getDiscriminator(type.innerType());
  } else if (type instanceof ZodLiteral) {
    return [type.value];
  } else if (type instanceof ZodEnum) {
    return type.options;
  } else if (type instanceof ZodNativeEnum) {
    return util.objectValues(type.enum);
  } else if (type instanceof ZodDefault) {
    return getDiscriminator(type._def.innerType);
  } else if (type instanceof ZodUndefined) {
    return [void 0];
  } else if (type instanceof ZodNull) {
    return [null];
  } else if (type instanceof ZodOptional) {
    return [void 0, ...getDiscriminator(type.unwrap())];
  } else if (type instanceof ZodNullable) {
    return [null, ...getDiscriminator(type.unwrap())];
  } else if (type instanceof ZodBranded) {
    return getDiscriminator(type.unwrap());
  } else if (type instanceof ZodReadonly) {
    return getDiscriminator(type.unwrap());
  } else if (type instanceof ZodCatch) {
    return getDiscriminator(type._def.innerType);
  } else {
    return [];
  }
};
var ZodDiscriminatedUnion = class _ZodDiscriminatedUnion extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.object) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.object,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const discriminator = this.discriminator;
    const discriminatorValue = ctx.data[discriminator];
    const option = this.optionsMap.get(discriminatorValue);
    if (!option) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_union_discriminator,
        options: Array.from(this.optionsMap.keys()),
        path: [discriminator]
      });
      return INVALID;
    }
    if (ctx.common.async) {
      return option._parseAsync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      });
    } else {
      return option._parseSync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      });
    }
  }
  get discriminator() {
    return this._def.discriminator;
  }
  get options() {
    return this._def.options;
  }
  get optionsMap() {
    return this._def.optionsMap;
  }
  /**
   * The constructor of the discriminated union schema. Its behaviour is very similar to that of the normal z.union() constructor.
   * However, it only allows a union of objects, all of which need to share a discriminator property. This property must
   * have a different value for each object in the union.
   * @param discriminator the name of the discriminator property
   * @param types an array of object schemas
   * @param params
   */
  static create(discriminator, options, params) {
    const optionsMap = /* @__PURE__ */ new Map();
    for (const type of options) {
      const discriminatorValues = getDiscriminator(type.shape[discriminator]);
      if (!discriminatorValues.length) {
        throw new Error(`A discriminator value for key \`${discriminator}\` could not be extracted from all schema options`);
      }
      for (const value of discriminatorValues) {
        if (optionsMap.has(value)) {
          throw new Error(`Discriminator property ${String(discriminator)} has duplicate value ${String(value)}`);
        }
        optionsMap.set(value, type);
      }
    }
    return new _ZodDiscriminatedUnion({
      typeName: ZodFirstPartyTypeKind.ZodDiscriminatedUnion,
      discriminator,
      options,
      optionsMap,
      ...processCreateParams(params)
    });
  }
};
function mergeValues(a, b) {
  const aType = getParsedType(a);
  const bType = getParsedType(b);
  if (a === b) {
    return { valid: true, data: a };
  } else if (aType === ZodParsedType.object && bType === ZodParsedType.object) {
    const bKeys = util.objectKeys(b);
    const sharedKeys = util.objectKeys(a).filter((key) => bKeys.indexOf(key) !== -1);
    const newObj = { ...a, ...b };
    for (const key of sharedKeys) {
      const sharedValue = mergeValues(a[key], b[key]);
      if (!sharedValue.valid) {
        return { valid: false };
      }
      newObj[key] = sharedValue.data;
    }
    return { valid: true, data: newObj };
  } else if (aType === ZodParsedType.array && bType === ZodParsedType.array) {
    if (a.length !== b.length) {
      return { valid: false };
    }
    const newArray = [];
    for (let index = 0; index < a.length; index++) {
      const itemA = a[index];
      const itemB = b[index];
      const sharedValue = mergeValues(itemA, itemB);
      if (!sharedValue.valid) {
        return { valid: false };
      }
      newArray.push(sharedValue.data);
    }
    return { valid: true, data: newArray };
  } else if (aType === ZodParsedType.date && bType === ZodParsedType.date && +a === +b) {
    return { valid: true, data: a };
  } else {
    return { valid: false };
  }
}
var ZodIntersection = class extends ZodType {
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    const handleParsed = (parsedLeft, parsedRight) => {
      if (isAborted(parsedLeft) || isAborted(parsedRight)) {
        return INVALID;
      }
      const merged = mergeValues(parsedLeft.value, parsedRight.value);
      if (!merged.valid) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.invalid_intersection_types
        });
        return INVALID;
      }
      if (isDirty(parsedLeft) || isDirty(parsedRight)) {
        status.dirty();
      }
      return { status: status.value, value: merged.data };
    };
    if (ctx.common.async) {
      return Promise.all([
        this._def.left._parseAsync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        }),
        this._def.right._parseAsync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        })
      ]).then(([left, right]) => handleParsed(left, right));
    } else {
      return handleParsed(this._def.left._parseSync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      }), this._def.right._parseSync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      }));
    }
  }
};
ZodIntersection.create = (left, right, params) => {
  return new ZodIntersection({
    left,
    right,
    typeName: ZodFirstPartyTypeKind.ZodIntersection,
    ...processCreateParams(params)
  });
};
var ZodTuple = class _ZodTuple extends ZodType {
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.array) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.array,
        received: ctx.parsedType
      });
      return INVALID;
    }
    if (ctx.data.length < this._def.items.length) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.too_small,
        minimum: this._def.items.length,
        inclusive: true,
        exact: false,
        type: "array"
      });
      return INVALID;
    }
    const rest = this._def.rest;
    if (!rest && ctx.data.length > this._def.items.length) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.too_big,
        maximum: this._def.items.length,
        inclusive: true,
        exact: false,
        type: "array"
      });
      status.dirty();
    }
    const items = [...ctx.data].map((item, itemIndex) => {
      const schema = this._def.items[itemIndex] || this._def.rest;
      if (!schema)
        return null;
      return schema._parse(new ParseInputLazyPath(ctx, item, ctx.path, itemIndex));
    }).filter((x) => !!x);
    if (ctx.common.async) {
      return Promise.all(items).then((results) => {
        return ParseStatus.mergeArray(status, results);
      });
    } else {
      return ParseStatus.mergeArray(status, items);
    }
  }
  get items() {
    return this._def.items;
  }
  rest(rest) {
    return new _ZodTuple({
      ...this._def,
      rest
    });
  }
};
ZodTuple.create = (schemas, params) => {
  if (!Array.isArray(schemas)) {
    throw new Error("You must pass an array of schemas to z.tuple([ ... ])");
  }
  return new ZodTuple({
    items: schemas,
    typeName: ZodFirstPartyTypeKind.ZodTuple,
    rest: null,
    ...processCreateParams(params)
  });
};
var ZodRecord = class _ZodRecord extends ZodType {
  get keySchema() {
    return this._def.keyType;
  }
  get valueSchema() {
    return this._def.valueType;
  }
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.object) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.object,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const pairs = [];
    const keyType = this._def.keyType;
    const valueType = this._def.valueType;
    for (const key in ctx.data) {
      pairs.push({
        key: keyType._parse(new ParseInputLazyPath(ctx, key, ctx.path, key)),
        value: valueType._parse(new ParseInputLazyPath(ctx, ctx.data[key], ctx.path, key)),
        alwaysSet: key in ctx.data
      });
    }
    if (ctx.common.async) {
      return ParseStatus.mergeObjectAsync(status, pairs);
    } else {
      return ParseStatus.mergeObjectSync(status, pairs);
    }
  }
  get element() {
    return this._def.valueType;
  }
  static create(first, second, third) {
    if (second instanceof ZodType) {
      return new _ZodRecord({
        keyType: first,
        valueType: second,
        typeName: ZodFirstPartyTypeKind.ZodRecord,
        ...processCreateParams(third)
      });
    }
    return new _ZodRecord({
      keyType: ZodString.create(),
      valueType: first,
      typeName: ZodFirstPartyTypeKind.ZodRecord,
      ...processCreateParams(second)
    });
  }
};
var ZodMap = class extends ZodType {
  get keySchema() {
    return this._def.keyType;
  }
  get valueSchema() {
    return this._def.valueType;
  }
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.map) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.map,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const keyType = this._def.keyType;
    const valueType = this._def.valueType;
    const pairs = [...ctx.data.entries()].map(([key, value], index) => {
      return {
        key: keyType._parse(new ParseInputLazyPath(ctx, key, ctx.path, [index, "key"])),
        value: valueType._parse(new ParseInputLazyPath(ctx, value, ctx.path, [index, "value"]))
      };
    });
    if (ctx.common.async) {
      const finalMap = /* @__PURE__ */ new Map();
      return Promise.resolve().then(async () => {
        for (const pair of pairs) {
          const key = await pair.key;
          const value = await pair.value;
          if (key.status === "aborted" || value.status === "aborted") {
            return INVALID;
          }
          if (key.status === "dirty" || value.status === "dirty") {
            status.dirty();
          }
          finalMap.set(key.value, value.value);
        }
        return { status: status.value, value: finalMap };
      });
    } else {
      const finalMap = /* @__PURE__ */ new Map();
      for (const pair of pairs) {
        const key = pair.key;
        const value = pair.value;
        if (key.status === "aborted" || value.status === "aborted") {
          return INVALID;
        }
        if (key.status === "dirty" || value.status === "dirty") {
          status.dirty();
        }
        finalMap.set(key.value, value.value);
      }
      return { status: status.value, value: finalMap };
    }
  }
};
ZodMap.create = (keyType, valueType, params) => {
  return new ZodMap({
    valueType,
    keyType,
    typeName: ZodFirstPartyTypeKind.ZodMap,
    ...processCreateParams(params)
  });
};
var ZodSet = class _ZodSet extends ZodType {
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.set) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.set,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const def = this._def;
    if (def.minSize !== null) {
      if (ctx.data.size < def.minSize.value) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.too_small,
          minimum: def.minSize.value,
          type: "set",
          inclusive: true,
          exact: false,
          message: def.minSize.message
        });
        status.dirty();
      }
    }
    if (def.maxSize !== null) {
      if (ctx.data.size > def.maxSize.value) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.too_big,
          maximum: def.maxSize.value,
          type: "set",
          inclusive: true,
          exact: false,
          message: def.maxSize.message
        });
        status.dirty();
      }
    }
    const valueType = this._def.valueType;
    function finalizeSet(elements2) {
      const parsedSet = /* @__PURE__ */ new Set();
      for (const element of elements2) {
        if (element.status === "aborted")
          return INVALID;
        if (element.status === "dirty")
          status.dirty();
        parsedSet.add(element.value);
      }
      return { status: status.value, value: parsedSet };
    }
    const elements = [...ctx.data.values()].map((item, i) => valueType._parse(new ParseInputLazyPath(ctx, item, ctx.path, i)));
    if (ctx.common.async) {
      return Promise.all(elements).then((elements2) => finalizeSet(elements2));
    } else {
      return finalizeSet(elements);
    }
  }
  min(minSize, message) {
    return new _ZodSet({
      ...this._def,
      minSize: { value: minSize, message: errorUtil.toString(message) }
    });
  }
  max(maxSize, message) {
    return new _ZodSet({
      ...this._def,
      maxSize: { value: maxSize, message: errorUtil.toString(message) }
    });
  }
  size(size, message) {
    return this.min(size, message).max(size, message);
  }
  nonempty(message) {
    return this.min(1, message);
  }
};
ZodSet.create = (valueType, params) => {
  return new ZodSet({
    valueType,
    minSize: null,
    maxSize: null,
    typeName: ZodFirstPartyTypeKind.ZodSet,
    ...processCreateParams(params)
  });
};
var ZodFunction = class _ZodFunction extends ZodType {
  constructor() {
    super(...arguments);
    this.validate = this.implement;
  }
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.function) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.function,
        received: ctx.parsedType
      });
      return INVALID;
    }
    function makeArgsIssue(args, error) {
      return makeIssue({
        data: args,
        path: ctx.path,
        errorMaps: [ctx.common.contextualErrorMap, ctx.schemaErrorMap, getErrorMap(), en_default].filter((x) => !!x),
        issueData: {
          code: ZodIssueCode.invalid_arguments,
          argumentsError: error
        }
      });
    }
    function makeReturnsIssue(returns, error) {
      return makeIssue({
        data: returns,
        path: ctx.path,
        errorMaps: [ctx.common.contextualErrorMap, ctx.schemaErrorMap, getErrorMap(), en_default].filter((x) => !!x),
        issueData: {
          code: ZodIssueCode.invalid_return_type,
          returnTypeError: error
        }
      });
    }
    const params = { errorMap: ctx.common.contextualErrorMap };
    const fn = ctx.data;
    if (this._def.returns instanceof ZodPromise) {
      const me = this;
      return OK(async function(...args) {
        const error = new ZodError([]);
        const parsedArgs = await me._def.args.parseAsync(args, params).catch((e) => {
          error.addIssue(makeArgsIssue(args, e));
          throw error;
        });
        const result = await Reflect.apply(fn, this, parsedArgs);
        const parsedReturns = await me._def.returns._def.type.parseAsync(result, params).catch((e) => {
          error.addIssue(makeReturnsIssue(result, e));
          throw error;
        });
        return parsedReturns;
      });
    } else {
      const me = this;
      return OK(function(...args) {
        const parsedArgs = me._def.args.safeParse(args, params);
        if (!parsedArgs.success) {
          throw new ZodError([makeArgsIssue(args, parsedArgs.error)]);
        }
        const result = Reflect.apply(fn, this, parsedArgs.data);
        const parsedReturns = me._def.returns.safeParse(result, params);
        if (!parsedReturns.success) {
          throw new ZodError([makeReturnsIssue(result, parsedReturns.error)]);
        }
        return parsedReturns.data;
      });
    }
  }
  parameters() {
    return this._def.args;
  }
  returnType() {
    return this._def.returns;
  }
  args(...items) {
    return new _ZodFunction({
      ...this._def,
      args: ZodTuple.create(items).rest(ZodUnknown.create())
    });
  }
  returns(returnType) {
    return new _ZodFunction({
      ...this._def,
      returns: returnType
    });
  }
  implement(func) {
    const validatedFunc = this.parse(func);
    return validatedFunc;
  }
  strictImplement(func) {
    const validatedFunc = this.parse(func);
    return validatedFunc;
  }
  static create(args, returns, params) {
    return new _ZodFunction({
      args: args ? args : ZodTuple.create([]).rest(ZodUnknown.create()),
      returns: returns || ZodUnknown.create(),
      typeName: ZodFirstPartyTypeKind.ZodFunction,
      ...processCreateParams(params)
    });
  }
};
var ZodLazy = class extends ZodType {
  get schema() {
    return this._def.getter();
  }
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    const lazySchema = this._def.getter();
    return lazySchema._parse({ data: ctx.data, path: ctx.path, parent: ctx });
  }
};
ZodLazy.create = (getter, params) => {
  return new ZodLazy({
    getter,
    typeName: ZodFirstPartyTypeKind.ZodLazy,
    ...processCreateParams(params)
  });
};
var ZodLiteral = class extends ZodType {
  _parse(input) {
    if (input.data !== this._def.value) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        received: ctx.data,
        code: ZodIssueCode.invalid_literal,
        expected: this._def.value
      });
      return INVALID;
    }
    return { status: "valid", value: input.data };
  }
  get value() {
    return this._def.value;
  }
};
ZodLiteral.create = (value, params) => {
  return new ZodLiteral({
    value,
    typeName: ZodFirstPartyTypeKind.ZodLiteral,
    ...processCreateParams(params)
  });
};
function createZodEnum(values, params) {
  return new ZodEnum({
    values,
    typeName: ZodFirstPartyTypeKind.ZodEnum,
    ...processCreateParams(params)
  });
}
var ZodEnum = class _ZodEnum extends ZodType {
  _parse(input) {
    if (typeof input.data !== "string") {
      const ctx = this._getOrReturnCtx(input);
      const expectedValues = this._def.values;
      addIssueToContext(ctx, {
        expected: util.joinValues(expectedValues),
        received: ctx.parsedType,
        code: ZodIssueCode.invalid_type
      });
      return INVALID;
    }
    if (!this._cache) {
      this._cache = new Set(this._def.values);
    }
    if (!this._cache.has(input.data)) {
      const ctx = this._getOrReturnCtx(input);
      const expectedValues = this._def.values;
      addIssueToContext(ctx, {
        received: ctx.data,
        code: ZodIssueCode.invalid_enum_value,
        options: expectedValues
      });
      return INVALID;
    }
    return OK(input.data);
  }
  get options() {
    return this._def.values;
  }
  get enum() {
    const enumValues = {};
    for (const val of this._def.values) {
      enumValues[val] = val;
    }
    return enumValues;
  }
  get Values() {
    const enumValues = {};
    for (const val of this._def.values) {
      enumValues[val] = val;
    }
    return enumValues;
  }
  get Enum() {
    const enumValues = {};
    for (const val of this._def.values) {
      enumValues[val] = val;
    }
    return enumValues;
  }
  extract(values, newDef = this._def) {
    return _ZodEnum.create(values, {
      ...this._def,
      ...newDef
    });
  }
  exclude(values, newDef = this._def) {
    return _ZodEnum.create(this.options.filter((opt) => !values.includes(opt)), {
      ...this._def,
      ...newDef
    });
  }
};
ZodEnum.create = createZodEnum;
var ZodNativeEnum = class extends ZodType {
  _parse(input) {
    const nativeEnumValues = util.getValidEnumValues(this._def.values);
    const ctx = this._getOrReturnCtx(input);
    if (ctx.parsedType !== ZodParsedType.string && ctx.parsedType !== ZodParsedType.number) {
      const expectedValues = util.objectValues(nativeEnumValues);
      addIssueToContext(ctx, {
        expected: util.joinValues(expectedValues),
        received: ctx.parsedType,
        code: ZodIssueCode.invalid_type
      });
      return INVALID;
    }
    if (!this._cache) {
      this._cache = new Set(util.getValidEnumValues(this._def.values));
    }
    if (!this._cache.has(input.data)) {
      const expectedValues = util.objectValues(nativeEnumValues);
      addIssueToContext(ctx, {
        received: ctx.data,
        code: ZodIssueCode.invalid_enum_value,
        options: expectedValues
      });
      return INVALID;
    }
    return OK(input.data);
  }
  get enum() {
    return this._def.values;
  }
};
ZodNativeEnum.create = (values, params) => {
  return new ZodNativeEnum({
    values,
    typeName: ZodFirstPartyTypeKind.ZodNativeEnum,
    ...processCreateParams(params)
  });
};
var ZodPromise = class extends ZodType {
  unwrap() {
    return this._def.type;
  }
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.promise && ctx.common.async === false) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.promise,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const promisified = ctx.parsedType === ZodParsedType.promise ? ctx.data : Promise.resolve(ctx.data);
    return OK(promisified.then((data) => {
      return this._def.type.parseAsync(data, {
        path: ctx.path,
        errorMap: ctx.common.contextualErrorMap
      });
    }));
  }
};
ZodPromise.create = (schema, params) => {
  return new ZodPromise({
    type: schema,
    typeName: ZodFirstPartyTypeKind.ZodPromise,
    ...processCreateParams(params)
  });
};
var ZodEffects = class extends ZodType {
  innerType() {
    return this._def.schema;
  }
  sourceType() {
    return this._def.schema._def.typeName === ZodFirstPartyTypeKind.ZodEffects ? this._def.schema.sourceType() : this._def.schema;
  }
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    const effect = this._def.effect || null;
    const checkCtx = {
      addIssue: (arg) => {
        addIssueToContext(ctx, arg);
        if (arg.fatal) {
          status.abort();
        } else {
          status.dirty();
        }
      },
      get path() {
        return ctx.path;
      }
    };
    checkCtx.addIssue = checkCtx.addIssue.bind(checkCtx);
    if (effect.type === "preprocess") {
      const processed = effect.transform(ctx.data, checkCtx);
      if (ctx.common.async) {
        return Promise.resolve(processed).then(async (processed2) => {
          if (status.value === "aborted")
            return INVALID;
          const result = await this._def.schema._parseAsync({
            data: processed2,
            path: ctx.path,
            parent: ctx
          });
          if (result.status === "aborted")
            return INVALID;
          if (result.status === "dirty")
            return DIRTY(result.value);
          if (status.value === "dirty")
            return DIRTY(result.value);
          return result;
        });
      } else {
        if (status.value === "aborted")
          return INVALID;
        const result = this._def.schema._parseSync({
          data: processed,
          path: ctx.path,
          parent: ctx
        });
        if (result.status === "aborted")
          return INVALID;
        if (result.status === "dirty")
          return DIRTY(result.value);
        if (status.value === "dirty")
          return DIRTY(result.value);
        return result;
      }
    }
    if (effect.type === "refinement") {
      const executeRefinement = (acc) => {
        const result = effect.refinement(acc, checkCtx);
        if (ctx.common.async) {
          return Promise.resolve(result);
        }
        if (result instanceof Promise) {
          throw new Error("Async refinement encountered during synchronous parse operation. Use .parseAsync instead.");
        }
        return acc;
      };
      if (ctx.common.async === false) {
        const inner = this._def.schema._parseSync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        });
        if (inner.status === "aborted")
          return INVALID;
        if (inner.status === "dirty")
          status.dirty();
        executeRefinement(inner.value);
        return { status: status.value, value: inner.value };
      } else {
        return this._def.schema._parseAsync({ data: ctx.data, path: ctx.path, parent: ctx }).then((inner) => {
          if (inner.status === "aborted")
            return INVALID;
          if (inner.status === "dirty")
            status.dirty();
          return executeRefinement(inner.value).then(() => {
            return { status: status.value, value: inner.value };
          });
        });
      }
    }
    if (effect.type === "transform") {
      if (ctx.common.async === false) {
        const base = this._def.schema._parseSync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        });
        if (!isValid(base))
          return INVALID;
        const result = effect.transform(base.value, checkCtx);
        if (result instanceof Promise) {
          throw new Error(`Asynchronous transform encountered during synchronous parse operation. Use .parseAsync instead.`);
        }
        return { status: status.value, value: result };
      } else {
        return this._def.schema._parseAsync({ data: ctx.data, path: ctx.path, parent: ctx }).then((base) => {
          if (!isValid(base))
            return INVALID;
          return Promise.resolve(effect.transform(base.value, checkCtx)).then((result) => ({
            status: status.value,
            value: result
          }));
        });
      }
    }
    util.assertNever(effect);
  }
};
ZodEffects.create = (schema, effect, params) => {
  return new ZodEffects({
    schema,
    typeName: ZodFirstPartyTypeKind.ZodEffects,
    effect,
    ...processCreateParams(params)
  });
};
ZodEffects.createWithPreprocess = (preprocess, schema, params) => {
  return new ZodEffects({
    schema,
    effect: { type: "preprocess", transform: preprocess },
    typeName: ZodFirstPartyTypeKind.ZodEffects,
    ...processCreateParams(params)
  });
};
var ZodOptional = class extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType === ZodParsedType.undefined) {
      return OK(void 0);
    }
    return this._def.innerType._parse(input);
  }
  unwrap() {
    return this._def.innerType;
  }
};
ZodOptional.create = (type, params) => {
  return new ZodOptional({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodOptional,
    ...processCreateParams(params)
  });
};
var ZodNullable = class extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType === ZodParsedType.null) {
      return OK(null);
    }
    return this._def.innerType._parse(input);
  }
  unwrap() {
    return this._def.innerType;
  }
};
ZodNullable.create = (type, params) => {
  return new ZodNullable({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodNullable,
    ...processCreateParams(params)
  });
};
var ZodDefault = class extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    let data = ctx.data;
    if (ctx.parsedType === ZodParsedType.undefined) {
      data = this._def.defaultValue();
    }
    return this._def.innerType._parse({
      data,
      path: ctx.path,
      parent: ctx
    });
  }
  removeDefault() {
    return this._def.innerType;
  }
};
ZodDefault.create = (type, params) => {
  return new ZodDefault({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodDefault,
    defaultValue: typeof params.default === "function" ? params.default : () => params.default,
    ...processCreateParams(params)
  });
};
var ZodCatch = class extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    const newCtx = {
      ...ctx,
      common: {
        ...ctx.common,
        issues: []
      }
    };
    const result = this._def.innerType._parse({
      data: newCtx.data,
      path: newCtx.path,
      parent: {
        ...newCtx
      }
    });
    if (isAsync(result)) {
      return result.then((result2) => {
        return {
          status: "valid",
          value: result2.status === "valid" ? result2.value : this._def.catchValue({
            get error() {
              return new ZodError(newCtx.common.issues);
            },
            input: newCtx.data
          })
        };
      });
    } else {
      return {
        status: "valid",
        value: result.status === "valid" ? result.value : this._def.catchValue({
          get error() {
            return new ZodError(newCtx.common.issues);
          },
          input: newCtx.data
        })
      };
    }
  }
  removeCatch() {
    return this._def.innerType;
  }
};
ZodCatch.create = (type, params) => {
  return new ZodCatch({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodCatch,
    catchValue: typeof params.catch === "function" ? params.catch : () => params.catch,
    ...processCreateParams(params)
  });
};
var ZodNaN = class extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.nan) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.nan,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return { status: "valid", value: input.data };
  }
};
ZodNaN.create = (params) => {
  return new ZodNaN({
    typeName: ZodFirstPartyTypeKind.ZodNaN,
    ...processCreateParams(params)
  });
};
var BRAND = /* @__PURE__ */ Symbol("zod_brand");
var ZodBranded = class extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    const data = ctx.data;
    return this._def.type._parse({
      data,
      path: ctx.path,
      parent: ctx
    });
  }
  unwrap() {
    return this._def.type;
  }
};
var ZodPipeline = class _ZodPipeline extends ZodType {
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.common.async) {
      const handleAsync = async () => {
        const inResult = await this._def.in._parseAsync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        });
        if (inResult.status === "aborted")
          return INVALID;
        if (inResult.status === "dirty") {
          status.dirty();
          return DIRTY(inResult.value);
        } else {
          return this._def.out._parseAsync({
            data: inResult.value,
            path: ctx.path,
            parent: ctx
          });
        }
      };
      return handleAsync();
    } else {
      const inResult = this._def.in._parseSync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      });
      if (inResult.status === "aborted")
        return INVALID;
      if (inResult.status === "dirty") {
        status.dirty();
        return {
          status: "dirty",
          value: inResult.value
        };
      } else {
        return this._def.out._parseSync({
          data: inResult.value,
          path: ctx.path,
          parent: ctx
        });
      }
    }
  }
  static create(a, b) {
    return new _ZodPipeline({
      in: a,
      out: b,
      typeName: ZodFirstPartyTypeKind.ZodPipeline
    });
  }
};
var ZodReadonly = class extends ZodType {
  _parse(input) {
    const result = this._def.innerType._parse(input);
    const freeze = (data) => {
      if (isValid(data)) {
        data.value = Object.freeze(data.value);
      }
      return data;
    };
    return isAsync(result) ? result.then((data) => freeze(data)) : freeze(result);
  }
  unwrap() {
    return this._def.innerType;
  }
};
ZodReadonly.create = (type, params) => {
  return new ZodReadonly({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodReadonly,
    ...processCreateParams(params)
  });
};
function cleanParams(params, data) {
  const p = typeof params === "function" ? params(data) : typeof params === "string" ? { message: params } : params;
  const p2 = typeof p === "string" ? { message: p } : p;
  return p2;
}
function custom(check, _params = {}, fatal) {
  if (check)
    return ZodAny.create().superRefine((data, ctx) => {
      const r = check(data);
      if (r instanceof Promise) {
        return r.then((r2) => {
          if (!r2) {
            const params = cleanParams(_params, data);
            const _fatal = params.fatal ?? fatal ?? true;
            ctx.addIssue({ code: "custom", ...params, fatal: _fatal });
          }
        });
      }
      if (!r) {
        const params = cleanParams(_params, data);
        const _fatal = params.fatal ?? fatal ?? true;
        ctx.addIssue({ code: "custom", ...params, fatal: _fatal });
      }
      return;
    });
  return ZodAny.create();
}
var late = {
  object: ZodObject.lazycreate
};
var ZodFirstPartyTypeKind;
(function(ZodFirstPartyTypeKind2) {
  ZodFirstPartyTypeKind2["ZodString"] = "ZodString";
  ZodFirstPartyTypeKind2["ZodNumber"] = "ZodNumber";
  ZodFirstPartyTypeKind2["ZodNaN"] = "ZodNaN";
  ZodFirstPartyTypeKind2["ZodBigInt"] = "ZodBigInt";
  ZodFirstPartyTypeKind2["ZodBoolean"] = "ZodBoolean";
  ZodFirstPartyTypeKind2["ZodDate"] = "ZodDate";
  ZodFirstPartyTypeKind2["ZodSymbol"] = "ZodSymbol";
  ZodFirstPartyTypeKind2["ZodUndefined"] = "ZodUndefined";
  ZodFirstPartyTypeKind2["ZodNull"] = "ZodNull";
  ZodFirstPartyTypeKind2["ZodAny"] = "ZodAny";
  ZodFirstPartyTypeKind2["ZodUnknown"] = "ZodUnknown";
  ZodFirstPartyTypeKind2["ZodNever"] = "ZodNever";
  ZodFirstPartyTypeKind2["ZodVoid"] = "ZodVoid";
  ZodFirstPartyTypeKind2["ZodArray"] = "ZodArray";
  ZodFirstPartyTypeKind2["ZodObject"] = "ZodObject";
  ZodFirstPartyTypeKind2["ZodUnion"] = "ZodUnion";
  ZodFirstPartyTypeKind2["ZodDiscriminatedUnion"] = "ZodDiscriminatedUnion";
  ZodFirstPartyTypeKind2["ZodIntersection"] = "ZodIntersection";
  ZodFirstPartyTypeKind2["ZodTuple"] = "ZodTuple";
  ZodFirstPartyTypeKind2["ZodRecord"] = "ZodRecord";
  ZodFirstPartyTypeKind2["ZodMap"] = "ZodMap";
  ZodFirstPartyTypeKind2["ZodSet"] = "ZodSet";
  ZodFirstPartyTypeKind2["ZodFunction"] = "ZodFunction";
  ZodFirstPartyTypeKind2["ZodLazy"] = "ZodLazy";
  ZodFirstPartyTypeKind2["ZodLiteral"] = "ZodLiteral";
  ZodFirstPartyTypeKind2["ZodEnum"] = "ZodEnum";
  ZodFirstPartyTypeKind2["ZodEffects"] = "ZodEffects";
  ZodFirstPartyTypeKind2["ZodNativeEnum"] = "ZodNativeEnum";
  ZodFirstPartyTypeKind2["ZodOptional"] = "ZodOptional";
  ZodFirstPartyTypeKind2["ZodNullable"] = "ZodNullable";
  ZodFirstPartyTypeKind2["ZodDefault"] = "ZodDefault";
  ZodFirstPartyTypeKind2["ZodCatch"] = "ZodCatch";
  ZodFirstPartyTypeKind2["ZodPromise"] = "ZodPromise";
  ZodFirstPartyTypeKind2["ZodBranded"] = "ZodBranded";
  ZodFirstPartyTypeKind2["ZodPipeline"] = "ZodPipeline";
  ZodFirstPartyTypeKind2["ZodReadonly"] = "ZodReadonly";
})(ZodFirstPartyTypeKind || (ZodFirstPartyTypeKind = {}));
var instanceOfType = (cls, params = {
  message: `Input not instance of ${cls.name}`
}) => custom((data) => data instanceof cls, params);
var stringType = ZodString.create;
var numberType = ZodNumber.create;
var nanType = ZodNaN.create;
var bigIntType = ZodBigInt.create;
var booleanType = ZodBoolean.create;
var dateType = ZodDate.create;
var symbolType = ZodSymbol.create;
var undefinedType = ZodUndefined.create;
var nullType = ZodNull.create;
var anyType = ZodAny.create;
var unknownType = ZodUnknown.create;
var neverType = ZodNever.create;
var voidType = ZodVoid.create;
var arrayType = ZodArray.create;
var objectType = ZodObject.create;
var strictObjectType = ZodObject.strictCreate;
var unionType = ZodUnion.create;
var discriminatedUnionType = ZodDiscriminatedUnion.create;
var intersectionType = ZodIntersection.create;
var tupleType = ZodTuple.create;
var recordType = ZodRecord.create;
var mapType = ZodMap.create;
var setType = ZodSet.create;
var functionType = ZodFunction.create;
var lazyType = ZodLazy.create;
var literalType = ZodLiteral.create;
var enumType = ZodEnum.create;
var nativeEnumType = ZodNativeEnum.create;
var promiseType = ZodPromise.create;
var effectsType = ZodEffects.create;
var optionalType = ZodOptional.create;
var nullableType = ZodNullable.create;
var preprocessType = ZodEffects.createWithPreprocess;
var pipelineType = ZodPipeline.create;
var ostring = () => stringType().optional();
var onumber = () => numberType().optional();
var oboolean = () => booleanType().optional();
var coerce = {
  string: ((arg) => ZodString.create({ ...arg, coerce: true })),
  number: ((arg) => ZodNumber.create({ ...arg, coerce: true })),
  boolean: ((arg) => ZodBoolean.create({
    ...arg,
    coerce: true
  })),
  bigint: ((arg) => ZodBigInt.create({ ...arg, coerce: true })),
  date: ((arg) => ZodDate.create({ ...arg, coerce: true }))
};
var NEVER = INVALID;

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/shopping/shelf-life.ts
var DAY = 864e5;
var DEFAULT_ZONE = "America/Los_Angeles";
function validDate(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = Date.parse(`${value}T00:00:00Z`);
  return Number.isFinite(parsed) && new Date(parsed).toISOString().slice(0, 10) === value;
}
function validZone(value) {
  try {
    new Intl.DateTimeFormat("en", { timeZone: value }).format(0);
    return true;
  } catch {
    return false;
  }
}
function localDay(now, timeZone) {
  const parts = new Intl.DateTimeFormat("en-US", { timeZone, year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(now);
  return ["year", "month", "day"].map((key) => parts.find((p) => p.type === key).value).join("-");
}
function shelfLifeExpiry(item, now = Date.now()) {
  const evidence = item.source?.shelfLife;
  const zone = evidence?.timeZone ?? DEFAULT_ZONE;
  const fallback = item.source?.expiry ?? { raw: null, date: null, status: "unknown", sourceField: null, sourceDate: null, retrievedAt: item.source?.retrievedAt ?? null, confidence: "unknown" };
  const valid = evidence && evidence.basis !== "unknown" && validDate(evidence.deadline) && validZone(zone);
  const expiry = evidence ? {
    raw: evidence.deadline,
    date: valid ? evidence.deadline : null,
    status: valid ? evidence.basis === "recorded" ? "date-recorded" : "estimate" : "unknown",
    sourceField: `${evidence.source}.deadline`,
    sourceDate: evidence.evaluatedOn,
    retrievedAt: evidence.retrievedAt,
    confidence: valid ? evidence.basis === "recorded" ? "source-recorded" : "estimated" : "unknown"
  } : fallback;
  const date2 = validDate(expiry.date) && validZone(zone) ? expiry.date : null;
  return { ...expiry, date: date2, daysUntil: date2 ? Math.round((Date.parse(`${date2}T00:00:00Z`) - Date.parse(`${localDay(now, zone)}T00:00:00Z`)) / DAY) : null, timeZone: zone };
}
function queryShelfLife(h, now = Date.now()) {
  if (!Number.isFinite(now)) throw new Error("Invalid shelf-life query time.");
  const items = [];
  let unknown = 0;
  let retrievedAt = null;
  const zones = /* @__PURE__ */ new Set();
  for (const item of h.kitchen) {
    const evidence = item.source?.shelfLife;
    const expiry = shelfLifeExpiry(item, now);
    const { date: deadline, daysUntil, timeZone: zone } = expiry;
    const basis = expiry.status === "date-recorded" ? "recorded" : "estimated";
    if (!deadline || daysUntil === null) {
      unknown++;
      continue;
    }
    zones.add(zone);
    const at = evidence?.retrievedAt ?? item.source?.expiry?.retrievedAt ?? item.source?.retrievedAt;
    if (at && (!retrievedAt || at > retrievedAt)) retrievedAt = at;
    if (daysUntil > 7) continue;
    const date2 = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", month: "short", day: "numeric" }).format(/* @__PURE__ */ new Date(`${deadline}T12:00:00Z`));
    items.push({
      kitchenId: item.id,
      deadline,
      daysUntil,
      basis,
      dateLabel: `${basis === "recorded" ? "Recorded" : "Estimated"} \xB7 ${date2}`,
      timingLabel: daysUntil < 0 ? `${Math.abs(daysUntil)} ${daysUntil === -1 ? "day" : "days"} past` : daysUntil === 0 ? "Today" : daysUntil === 1 ? "Tomorrow" : `In ${daysUntil} days`,
      status: daysUntil < 0 ? "past" : "nearing",
      reason: evidence?.reason ?? (basis === "recorded" ? "Recorded date" : "Date estimate from your kitchen")
    });
  }
  items.sort((a, b) => a.deadline.localeCompare(b.deadline) || a.kitchenId.localeCompare(b.kitchenId));
  const past = items.filter((item) => item.status === "past").length;
  return {
    items,
    counts: { past, nearing: items.length - past, unknown, total: items.length },
    totalKitchenItems: h.kitchen.length,
    asOf: new Date(now).toISOString(),
    timeZone: zones.size === 1 ? [...zones][0] : DEFAULT_ZONE,
    retrievedAt
  };
}
var civilSchema = external_exports.string().refine(validDate, "Invalid civil date");
var evidenceSchema = external_exports.object({
  policyVersion: external_exports.string().min(1).max(100),
  source: external_exports.enum(["trepo-shelf-policy", "trepo-shelf-lifecycle"]),
  projection: external_exports.enum(["read-only-local-projection", "stored-lifecycle"]),
  deadline: civilSchema.nullable(),
  basis: external_exports.enum(["recorded", "estimated", "unknown"]),
  recordedDate: civilSchema.nullable(),
  estimatedDate: civilSchema.nullable(),
  timeZone: external_exports.string().max(100).refine(validZone),
  evaluatedOn: civilSchema,
  retrievedAt: external_exports.string().datetime({ offset: true }),
  reason: external_exports.string().max(400),
  clockSource: external_exports.string().max(100).nullable()
}).strict().refine((value) => value.deadline === null === (value.basis === "unknown"), "Deadline basis mismatch");
var payloadSchema = external_exports.object({
  version: external_exports.literal(1),
  subjectActorId: external_exports.string().min(1),
  retrievedAt: external_exports.string().datetime({ offset: true }),
  completeActiveInventory: external_exports.literal(true).optional(),
  items: external_exports.array(external_exports.object({ sourceId: external_exports.string().min(1), shelfLife: evidenceSchema }).strict()).max(5e3)
}).strict();

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/shopping/kitchen-knowledge.ts
var DAY2 = 864e5;
var text = (v) => typeof v === "string" && v.trim() ? v.trim() : null;
var iso = (v) => typeof v === "string" && /^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$/.test(v) && Number.isFinite(Date.parse(v)) ? new Date(v).toISOString() : null;
var day = (value) => {
  if (!/^\d{4}-\d{2}-\d{2}(?:T.*(?:Z|[+-]\d{2}:\d{2}))?$/.test(value)) return null;
  const prefix = value.slice(0, 10), parsed = Date.parse(`${prefix}T00:00:00Z`);
  if (!Number.isFinite(parsed) || new Date(parsed).toISOString().slice(0, 10) !== prefix) return null;
  return value.length === 10 || iso(value) ? prefix : null;
};
function expiryEvidence(expiration, provenance, retrievedAt, sourceDate = null) {
  const raw = text(expiration), date2 = raw ? day(raw) : null;
  const unknown = { raw, date: null, status: "unknown", sourceField: raw ? "product_expiration" : null, sourceDate, retrievedAt, confidence: "unknown" };
  if (!date2) return unknown;
  const p = provenance && typeof provenance === "object" && !Array.isArray(provenance) ? provenance : null;
  const verified = p && p.value === raw && ["item_edit", "create_request", "expiration_image"].includes(String(p.source)) && iso(p.observed_at);
  return { raw, date: date2, status: verified ? "date-recorded" : "estimate", sourceField: verified ? `expiration_provenance.${String(p.source)}` : "product_expiration", sourceDate: verified ? iso(p.observed_at) : sourceDate, retrievedAt, confidence: verified ? "source-recorded" : "estimated" };
}
function observationActivity(o) {
  if (o.activity) return o.activity;
  if (o.sourceTable.includes("archive")) return { kind: "archive-snapshot", occurredAt: o.archivedAt ?? null, sourceField: "archived_at", rawAction: o.sourceState, reason: o.archiveReason ?? null, evidence: "source-state" };
  const kind = o.sourceState?.toUpperCase() === "IN" ? "inventory-in" : o.sourceState?.toUpperCase() === "OUT" ? "inventory-out" : "unclassified";
  return { kind, occurredAt: o.observedAt, sourceField: "action", rawAction: o.sourceState, reason: null, evidence: kind === "unclassified" ? "unknown" : "source-state" };
}
var nullableText = external_exports.string().max(300).nullable();
var eventSchema = external_exports.object({ id: external_exports.string().min(1).max(200), source: external_exports.enum(["amount-operation", "archive-reason", "feed-event", "retailer-order"]), sourceId: external_exports.string().min(1).max(200), operationId: nullableText, itemSourceId: nullableText, actorId: nullableText, occurredAt: external_exports.string().datetime({ offset: true }).nullable(), kind: external_exports.enum(["consumed", "discarded", "purchased", "inventory-in", "inventory-out", "unknown-removal"]), name: nullableText, brand: nullableText, quantity: external_exports.number().finite().nonnegative().nullable(), unit: nullableText, quantityBasis: external_exports.enum(["explicit-delta", "remaining-snapshot", "unknown"]), undoneBy: nullableText, sourceAction: external_exports.string().max(100), sourceLabel: external_exports.string().max(160) }).strict();
var payloadSchema2 = external_exports.object({ version: external_exports.literal(1), subjectActorId: external_exports.string().min(1), sourceLabel: external_exports.string().min(1).max(160), retrievedAt: external_exports.string().datetime({ offset: true }), kitchen: external_exports.array(external_exports.object({ sourceId: external_exports.string().min(1), expiration: nullableText, expirationProvenance: external_exports.object({ value: external_exports.string().max(300), source: external_exports.enum(["item_edit", "create_request", "expiration_image"]), observed_at: external_exports.string().datetime({ offset: true }) }).strict().nullable(), openedAt: external_exports.string().datetime({ offset: true }).nullable(), isOpened: external_exports.boolean().nullable() }).strict()).max(5e3), events: external_exports.array(eventSchema).max(2e4) }).strict();
function validEvent(e) {
  const action = e.sourceAction.toLowerCase();
  if (e.kind === "purchased") return e.source === "retailer-order" && ["confirmed", "accepted", "paid"].includes(action);
  if (e.kind === "consumed") return e.source === "amount-operation" && action === "consume" || e.source === "archive-reason" && ["consumed_manual", "consumed_recipe"].includes(action);
  if (e.kind === "discarded") return e.source === "amount-operation" && action === "discard" || e.source === "archive-reason" && action === "clean_discard" || e.source === "feed-event" && ["clean_discard", "discard_add"].includes(action);
  if (e.kind === "inventory-in") return e.source === "feed-event" && action === "kitchen_add";
  if (e.kind === "inventory-out") return e.source === "feed-event" && action === "kitchen_remove";
  if (e.kind === "unknown-removal") return ["archive-reason", "feed-event"].includes(e.source);
  return false;
}
function dedupeKitchenEvents(input, subjectActorId) {
  const result = { events: [], excludedUndone: 0, excludedOtherActors: 0, duplicatesExcluded: 0 };
  const personal = input.filter((e) => {
    if (e.actorId !== subjectActorId) {
      result.excludedOtherActors++;
      return false;
    }
    return true;
  });
  const operationKey = (e) => e.operationId && e.itemSourceId ? `${e.operationId}:${e.itemSourceId}` : null;
  const identity2 = (e) => e.itemSourceId && e.occurredAt ? `${e.itemSourceId}:${e.kind}:${iso(e.occurredAt)}` : null;
  const undone = new Set(personal.filter((e) => e.undoneBy).map(operationKey).filter(Boolean)), undoneIdentity = new Set(personal.filter((e) => e.undoneBy).map(identity2).filter(Boolean));
  const seen = /* @__PURE__ */ new Set(), ledgerIdentity = /* @__PURE__ */ new Set();
  for (const e of [...personal].sort((a, b) => Number(b.source === "amount-operation") - Number(a.source === "amount-operation"))) {
    if (!validEvent(e)) throw new Error("Knowledge event kind is not supported by its verified source action.");
    const op = operationKey(e), eventIdentity = identity2(e);
    if (e.undoneBy || op && undone.has(op) || e.source !== "amount-operation" && eventIdentity && undoneIdentity.has(eventIdentity)) {
      result.excludedUndone++;
      continue;
    }
    const key = op ?? `${e.source}:${e.sourceId}:${e.itemSourceId ?? ""}:${e.kind}`;
    if (seen.has(key) || e.source !== "amount-operation" && eventIdentity && ledgerIdentity.has(eventIdentity)) {
      result.duplicatesExcluded++;
      continue;
    }
    seen.add(key);
    if (e.source === "amount-operation" && eventIdentity) ledgerIdentity.add(eventIdentity);
    result.events.push(structuredClone(e));
  }
  return result;
}
function applyKitchenKnowledge(h, input, verifiedActorId, now = Date.now()) {
  const parsed = payloadSchema2.safeParse(input);
  if (!parsed.success) throw new Error("Invalid private kitchen knowledge payload.");
  const s = parsed.data;
  if (s.subjectActorId !== verifiedActorId || h.onlineShopping?.profile.subjectActorId !== verifiedActorId || h.kitchenKnowledge && h.kitchenKnowledge.subjectActorId !== verifiedActorId) throw new Error("Kitchen knowledge actor does not match the verified local profile.");
  if (Date.parse(s.retrievedAt) > now + 3e5 || now - Date.parse(s.retrievedAt) > DAY2) throw new Error("Refresh private kitchen knowledge before applying it.");
  if (new Set(s.kitchen.map((k) => k.sourceId)).size !== s.kitchen.length) throw new Error("Duplicate kitchen source IDs in knowledge payload.");
  const eventResult = dedupeKitchenEvents(s.events, verifiedActorId), next = structuredClone(h);
  let updatedKitchenItems = 0;
  for (const k of next.kitchen) {
    const record = s.kitchen.find((r) => r.sourceId === k.source?.originalId);
    if (!record || !k.source) continue;
    k.source.expiry = expiryEvidence(record.expiration, record.expirationProvenance, s.retrievedAt);
    k.source.openedAt = record.openedAt;
    k.source.isOpened = record.isOpened;
    updatedKitchenItems++;
  }
  next.kitchenKnowledge = { subjectActorId: verifiedActorId, sourceLabel: s.sourceLabel, retrievedAt: s.retrievedAt, completeHistory: false, ...eventResult };
  next.revision++;
  return { household: next, report: { updatedKitchenItems, unmatchedSourceItems: s.kitchen.length - updatedKitchenItems, events: eventResult.events.length, excludedUndone: eventResult.excludedUndone, excludedOtherActors: eventResult.excludedOtherActors, duplicatesExcluded: eventResult.duplicatesExcluded } };
}

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/shopping/gtin.ts
function normalizeGTIN(input) {
  const v = input.replace(/[\s-]/g, "");
  if (!/^\d+$/.test(v) || ![8, 12, 13, 14].includes(v.length)) return null;
  const digits = [...v].map(Number);
  const check = digits.pop();
  const sum = digits.reverse().reduce((s, n, i) => s + n * (i % 2 === 0 ? 3 : 1), 0);
  return (10 - sum % 10) % 10 === check ? v.padStart(14, "0") : null;
}

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/shopping/online.ts
var clean = (v) => typeof v === "string" || typeof v === "number" ? String(v).trim() || null : null;
var field = (r, ...keys) => keys.map((k) => clean(r[k])).find((v) => v !== null) ?? null;
var norm = (s) => (s ?? "").normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}%]+/gu, " ").trim();
var words = (s) => norm(s).replace(/\b(oat|almond|soy)milk\b/g, "$1 milk").split(" ").filter((w) => w && !["a", "an", "the", "of", "and"].includes(w)).map((w) => w.length > 4 && w.endsWith("s") ? w.slice(0, -1) : w);
var uncertain = (s) => !!s && /\b(likely|possibly|possible|unclear|uncertain|appears|may be|might be|guess|unknown)\b|\?/i.test(s);
var date = (s) => s && Number.isFinite(Date.parse(s)) ? new Date(s).toISOString() : null;
async function hash(s) {
  return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s))), (b) => b.toString(16).padStart(2, "0")).join("");
}
var historySchema = external_exports.object({
  version: external_exports.literal(1),
  subjectActorId: external_exports.string().min(1),
  sourceLabel: external_exports.string().min(1).max(160),
  retrievedAt: external_exports.string().datetime({ offset: true }),
  allowedHouseholdActorIds: external_exports.array(external_exports.string()).optional(),
  coverage: external_exports.array(external_exports.object({ table: external_exports.string(), count: external_exports.number() })).optional(),
  records: external_exports.array(external_exports.object({ sourceTable: external_exports.string().min(1), record: external_exports.record(external_exports.unknown()) }).strict()).max(2e4)
}).strict();
async function buildUserProfile(input, verifiedActorId, now = (/* @__PURE__ */ new Date()).toISOString()) {
  const parsed = historySchema.safeParse(input);
  if (!parsed.success) throw new Error("Invalid private history snapshot.");
  const s = parsed.data;
  if (s.subjectActorId !== verifiedActorId) throw new Error("History subject does not match the independently verified actor.");
  const report = { excludedOtherActors: 0, excludedUnattributed: 0, duplicates: 0 };
  const observations = /* @__PURE__ */ new Map();
  const priority = (table) => table === "shared_kitchen" ? 0 : table === "shared_archive_kitchen" ? 1 : table.includes("prod_kitchen") ? 2 : 3;
  for (const entry of [...s.records].sort((a, b) => priority(a.sourceTable) - priority(b.sourceTable))) {
    const r = entry.record, originalActor = field(r, "user_id", "actorId", "actor_id"), owner = field(r, "_owner", "ownerId", "owner_id");
    if (originalActor && originalActor !== verifiedActorId) {
      report.excludedOtherActors++;
      continue;
    }
    if (!originalActor && owner && owner !== verifiedActorId) {
      report.excludedOtherActors++;
      continue;
    }
    const accountScoped = owner === verifiedActorId || entry.sourceTable.includes(verifiedActorId);
    if (!originalActor && !accountScoped) {
      report.excludedUnattributed++;
      continue;
    }
    const recordId = field(r, "_id", "id", "itemUUID"), originalId = field(r, "original_id", "originalId") ?? recordId;
    const name = field(r, "product_name", "name", "title");
    if (!recordId || !originalId || !name) {
      report.excludedUnattributed++;
      continue;
    }
    const family = /^\d+$/.test(originalId) ? `${entry.sourceTable}:${originalId}` : originalId.toLowerCase();
    if (observations.has(family)) {
      report.duplicates++;
      continue;
    }
    const captureSource = field(r, "captureSource", "capture_source", "analysis_source", "source", "method", "input_type");
    const brand = field(r, "brand", "product_brand"), variant = field(r, "variant", "product_variant");
    const gtin = field(r, "barcode", "product_barcode", "barcode_number");
    observations.set(family, {
      id: `obs-${(await hash(`${verifiedActorId}:${family}`)).slice(0, 24)}`,
      subjectActorId: verifiedActorId,
      sourceRecordId: originalId,
      sourceTable: entry.sourceTable,
      sourceLabel: s.sourceLabel,
      retrievedAt: s.retrievedAt,
      observedAt: date(field(r, "_createdDate", "created_at", "createdDate", "created_date", "timestamp")),
      kind: /receipt/i.test(captureSource ?? "") ? "receipt-observation" : "kitchen-observation",
      sourceState: field(r, "action", "status", "analysis_status"),
      imageEmoji: ["resized_image_url", "product_image_url", "image_url", "image", "images"].map((k) => sourceEmoji(field(r, k) ?? "")).find(Boolean) ?? null,
      attribution: originalActor ? "recorded-actor" : "account-context",
      captureSource,
      name,
      brand,
      variant,
      size: field(r, "product_size", "package_size", "size"),
      gtin: gtin && normalizeGTIN(gtin) ? normalizeGTIN(gtin) : null,
      archivedAt: date(field(r, "archived_at")),
      archiveReason: field(r, "archived_reason"),
      ...r.product_expiration !== void 0 ? { expiry: expiryEvidence(r.product_expiration, r.expiration_provenance ?? r.overrides__expiration_provenance, s.retrievedAt, date(field(r, "_createdDate", "created_at"))) } : {},
      ...typeof r.quantity_value === "number" && Number.isFinite(r.quantity_value) ? { quantityValue: r.quantity_value, quantityUnit: field(r, "quantity_unit") } : {},
      category: field(r, "category", "aisle_category"),
      confidence: [name, brand, variant].some(uncertain) || entry.sourceTable.includes("archive") || r.needs_review === true || /preliminary|quick_identify/.test(captureSource ?? "") ? "uncertain" : "recorded",
      purchaseConfirmed: false
    });
  }
  const facts = /* @__PURE__ */ new Map();
  for (const o of observations.values()) {
    for (const [kind, value] of [["product-observation", o.name], ["brand-observation", o.brand]]) {
      if (!value || uncertain(value)) continue;
      const key = `${kind}:${norm(value)}${kind === "product-observation" ? `:${norm(o.brand)}:${norm(o.variant)}:${norm(o.size)}` : ""}`;
      let fact = facts.get(key);
      if (!fact) {
        fact = { id: `fact-${(await hash(`${verifiedActorId}:${key}`)).slice(0, 24)}`, subjectActorId: verifiedActorId, kind, value, productName: kind === "product-observation" ? o.name : null, brand: o.brand, occurrences: 0, firstSeenAt: null, lastSeenAt: null, confidence: "low", status: "inferred", attribution: o.attribution, evidenceIds: [] };
        facts.set(key, fact);
      }
      fact.occurrences++;
      fact.evidenceIds.push(o.id);
      if (fact.attribution !== o.attribution) fact.attribution = "mixed";
      if (o.observedAt) {
        if (!fact.firstSeenAt || o.observedAt < fact.firstSeenAt) fact.firstSeenAt = o.observedAt;
        if (!fact.lastSeenAt || o.observedAt > fact.lastSeenAt) fact.lastSeenAt = o.observedAt;
      }
    }
  }
  const byId = new Map([...observations.values()].map((o) => [o.id, o]));
  for (const fact of facts.values()) fact.confidence = fact.occurrences >= 3 && fact.attribution === "recorded-actor" && fact.evidenceIds.every((id) => byId.get(id)?.confidence === "recorded") ? "medium" : "low";
  return { profile: {
    id: `profile-${(await hash(verifiedActorId)).slice(0, 24)}`,
    subjectActorId: verifiedActorId,
    generatedAt: now,
    sourceLabel: s.sourceLabel,
    retrievedAt: s.retrievedAt,
    status: "local-inferred-profile",
    observationCount: observations.size,
    confirmedChoices: [],
    facts: [...facts.values()].sort((a, b) => b.occurrences - a.occurrences || a.id.localeCompare(b.id)),
    observations: [...observations.values()],
    caveat: "Kitchen and receipt observations do not confirm purchases or dietary preferences."
  }, report };
}
var productSchema = external_exports.object({
  id: external_exports.string().min(1).max(120),
  retailer: external_exports.enum(["instacart", "amazon", "safeway"]),
  sku: external_exports.string().min(1).max(160),
  name: external_exports.string().min(1).max(500),
  brand: external_exports.string().max(160).nullable(),
  variant: external_exports.string().max(200).nullable(),
  size: external_exports.string().max(160).nullable(),
  packCount: external_exports.number().int().min(1).max(100).nullable(),
  gtin: external_exports.string().nullable(),
  attributes: external_exports.array(external_exports.string().max(80)).max(40),
  allergens: external_exports.array(external_exports.string().max(80)).max(40).nullable(),
  image: external_exports.object({ url: external_exports.string().max(3e3).url().nullable(), sourceUrl: external_exports.string().max(3e3).url(), permission: external_exports.literal("remote-display") }).strict(),
  sourceUrl: external_exports.string().max(3e3).url(),
  retrievedAt: external_exports.string().datetime({ offset: true }),
  evidenceStatus: external_exports.enum(["verified-product-page", "search-result"]),
  priceCents: external_exports.number().int().min(0).max(1e6).nullable(),
  currency: external_exports.literal("USD"),
  priceScope: external_exports.enum(["public-page-unlocalized", "localized-page", "unknown"]),
  availability: external_exports.enum(["observed-available", "out-of-stock", "unknown"]),
  storeLabel: external_exports.string().max(100).optional(),
  observedZip: external_exports.string().regex(/^\d{5}$/).optional()
}).strict();
function candidate(item, p, profile, now) {
  const pref = item.preference, evidence = [p.name, p.brand, p.variant, p.size, ...p.attributes].filter(Boolean).join(" "), tokens = words(evidence);
  const request = words(item.name), missingTerms = request.filter((w) => !tokens.includes(w));
  const relatedTypes = [["dressing", "sauce", "marinade"]];
  const hardRequestTerms = /* @__PURE__ */ new Set(["without", "no", "avoid", "allergic", "allergy", "free", "gluten", "vegan", "vegetarian", "organic", "unsweetened", "sugar", "sodium", "decaf", "non", "gmo"]);
  const relatedVariation = missingTerms.length === 1 && !missingTerms.some((w) => hardRequestTerms.has(w)) && !request.some((w) => ["without", "no", "avoid", "allergic", "allergy"].includes(w)) && relatedTypes.some((group) => request.some((w) => group.includes(w)) && tokens.some((w) => group.includes(w))) && request.filter((w) => tokens.includes(w)).length >= 2;
  if (!request.length || missingTerms.length && !relatedVariation) return null;
  if (pref.requiredAttributes.some((a) => !p.attributes.some((v) => norm(v) === norm(a)))) return null;
  if (pref.excludedAllergens.length && (p.allergens === null || pref.excludedAllergens.some((a) => p.allergens.some((v) => norm(v) === norm(a))))) return null;
  const differences = missingTerms.length ? [`Requested terms not confirmed: ${missingTerms.join(", ")}`] : [];
  for (const [label, wanted, actual] of [["Brand", pref.brand, p.brand], ["Variety", pref.variant, p.variant], ["Size", pref.size, p.size]]) if (wanted && norm(wanted) !== norm(actual)) differences.push(`${label}: ${wanted} \u2192 ${actual ?? "not recorded"}`);
  if (pref.packCount !== null && pref.packCount !== p.packCount) differences.push(`Pack count: ${pref.packCount} \u2192 ${p.packCount ?? "not recorded"}`);
  if (pref.gtin && normalizeGTIN(pref.gtin) !== (p.gtin ? normalizeGTIN(p.gtin) : null)) differences.push("UPC / GTIN differs or is unavailable");
  const reasons = [missingTerms.length ? "Related product; review the differences" : "Matches the requested product terms"], uncertainties = [];
  let score = 50 + request.length * 4 - differences.length * 8;
  const neutralNameTerms = /* @__PURE__ */ new Set(["mountain", "natural", "artesian", "pure", "original", "premium", "fl", "oz", "ounce", "lb", "pound", "g", "kg", "ml", "l", "ct", "count", "pack", "pk", "can", "bottle", "case"]);
  const brandTerms = new Set(words(p.brand ?? ""));
  const extraNameTerms = new Set(words(p.name).filter((w) => !request.includes(w) && !brandTerms.has(w) && !neutralNameTerms.has(w) && !/^\d/.test(w)));
  score -= Math.min(6, extraNameTerms.size * 0.75);
  if (pref.gtin && p.gtin && normalizeGTIN(pref.gtin) === normalizeGTIN(p.gtin)) {
    score += 40;
    reasons.push("Matches the requested barcode");
  }
  if (pref.brand && norm(pref.brand) === norm(p.brand)) {
    score += 20;
    reasons.push("Matches the requested brand");
  }
  const brandFact = profile.facts.find((f) => f.kind === "brand-observation" && norm(f.value) === norm(p.brand));
  if (brandFact) {
    score += Math.min(8, Math.log2(brandFact.occurrences + 1) * 2);
    reasons.push(`Brand appears in ${brandFact.occurrences} kitchen observation${brandFact.occurrences === 1 ? "" : "s"}; inferred signal`);
  }
  const genericHistoryWords = /* @__PURE__ */ new Set(["organic", "fresh", "original", "dressing", "sauce", "marinade", "dipping", "with", "made", "milk", "water", "juice", "oil", "unspecified"]);
  const families = [["dressing", "sauce", "marinade"], ["milk"], ["water"], ["juice"], ["oil"], ["tampon"]];
  const family = (terms) => families.findIndex((group) => terms.some((w) => group.includes(w)));
  const candidateFamily = family(tokens), milkForms = ["oat", "almond", "soy", "coconut", "cashew", "rice", "goat", "cow"];
  const observations = new Map(profile.observations.map((o) => [o.id, o]));
  const related = profile.facts.filter((f) => f.kind === "product-observation" && (!f.brand || norm(f.brand) === norm(p.brand))).map((f) => {
    const observed = observations.get(f.evidenceIds[0]);
    const historicalTokens = words([f.value, observed?.variant].filter(Boolean).join(" "));
    const historyFamily = family(historicalTokens);
    const incompatibleFamily = historyFamily >= 0 && candidateFamily >= 0 && historyFamily !== candidateFamily;
    const incompatibleMilk = historyFamily === 1 && milkForms.some((w) => historicalTokens.includes(w) && !tokens.includes(w));
    const brandWords = new Set(words(f.brand ?? p.brand ?? ""));
    const details = [...new Set(historicalTokens.filter((w) => !genericHistoryWords.has(w) && !brandWords.has(w) && !/^\d/.test(w)))];
    const requested = details.filter((w) => request.includes(w)), shared = requested.filter((w) => tokens.includes(w));
    const sameBrand = !!f.brand && norm(f.brand) === norm(p.brand), specificMilk = historyFamily === 1 && milkForms.some((w) => shared.includes(w));
    const relevant = !incompatibleFamily && !incompatibleMilk && (requested.length >= 2 || sameBrand && requested.length >= 1 || specificMilk);
    const age = f.lastSeenAt ? (now - Date.parse(f.lastSeenAt)) / 864e5 : null;
    const recency = age === null ? 0.5 : age > 365 ? 0.5 : age > 180 ? 0.75 : 1;
    const weight = relevant && requested.length ? (shared.length * 6 + Math.min(2, Math.log2(f.occurrences + 1))) * (shared.length / requested.length) * recency * (f.confidence === "medium" ? 1 : 0.75) : 0;
    return { shared, weight };
  }).sort((a, b) => b.weight - a.weight)[0];
  if (related?.weight) {
    score += related.weight;
    reasons.splice(1, 0, `Kitchen history includes ${related.shared.slice(0, 3).join(" / ")} details; inferred, not a confirmed purchase`);
  }
  const choice = (profile.confirmedChoices ?? []).find((c) => norm(c.requestName) === norm(item.name) && c.productId === p.id);
  if (choice && !differences.length) {
    score += 100;
    reasons.unshift("You previously chose this product for this request; not a purchase record");
  }
  if (p.priceScope === "public-page-unlocalized") uncertainties.push("Public page price; delivery location and account pricing are not verified");
  if (p.priceCents === null) uncertainties.push("Price was not available in the source evidence");
  if (now - Date.parse(p.retrievedAt) > 864e5) uncertainties.push("Product evidence is over 24 hours old; recheck before creating a retailer cart");
  if (p.availability !== "observed-available") uncertainties.push(p.availability === "out-of-stock" ? "Source reported out of stock" : "Current availability is unknown");
  if (!p.size || !p.packCount) uncertainties.push("Package size or pack count is not fully recorded");
  if (!p.image.url) uncertainties.push("Source product image unavailable");
  if (p.evidenceStatus === "search-result") uncertainties.push("Only a search result was observed; product page needs verification");
  return { product: p, score: Math.round(score * 100) / 100, reasons, uncertainties, differences, selectable: p.evidenceStatus === "verified-product-page" && p.availability !== "out-of-stock" };
}
function onlineMatches(h, now = Date.now()) {
  const online = h.onlineShopping;
  if (!online) return [];
  return h.list.map((item) => {
    const candidates = online.catalog.map((p) => candidate(item, p, online.profile, now)).filter((c) => c !== null).sort((a, b) => Number(b.selectable) - Number(a.selectable) || b.score - a.score || a.product.id.localeCompare(b.product.id));
    const selection = online.selections[item.id], selected = candidates.find((c) => c.product.id === selection?.productId && c.selectable && (!c.differences.length || selection.differencesAccepted));
    const suggested = candidates.find((c) => c.selectable && !c.differences.length) ?? candidates.find((c) => c.selectable);
    return { itemId: item.id, requestedName: item.name, status: selected ? "selected" : suggested ? "suggested" : "blocked", suggestedProductId: suggested?.product.id ?? null, selectedProductId: selected?.product.id ?? null, candidates, reason: selected ? "Selected for this local draft" : suggested ? "Real product suggestion; retailer cart has not been created" : "No verified selectable product meets this request and its hard restrictions" };
  });
}
function attachOnlineShopping(h, profile, products, retrievedAt, now = Date.now()) {
  if (h.onlineShopping && h.onlineShopping.profile.subjectActorId !== profile.subjectActorId) throw new Error("Profile actor cannot change for an existing local copy.");
  const next = structuredClone(h);
  const state = { profile: { ...profile, confirmedChoices: h.onlineShopping?.profile.confirmedChoices ?? profile.confirmedChoices ?? [] }, catalog: products, catalogRetrievedAt: retrievedAt, selections: {}, draftCart: null, searchRequests: h.onlineShopping?.searchRequests ?? [] };
  for (const [itemId, selection] of Object.entries(h.onlineShopping?.selections ?? {})) {
    const before = h.onlineShopping.catalog.find((p) => p.id === selection.productId), after = products.find((p) => p.id === selection.productId);
    const identity2 = (p) => JSON.stringify([p.id, p.name, p.brand, p.variant, p.size, p.packCount, p.gtin, p.attributes, p.allergens]);
    if (before && after && identity2(before) === identity2(after)) state.selections[itemId] = selection;
  }
  next.onlineShopping = state;
  next.revision++;
  next.purchaseRevision++;
  reconcileOnlineSearches(next, now);
  return next;
}
function reconcileOnlineSearches(h, now = Date.now(), forceItemId) {
  const online = h.onlineShopping;
  if (!online) return;
  const existing = online.searchRequests ?? [], matches = onlineMatches(h, now);
  online.searchRequests = h.list.flatMap((item) => {
    const requestKey = JSON.stringify([item.id, item.name, item.preference]);
    const previous = existing.find((r) => r.itemId === item.id && r.requestKey === requestKey);
    const match = matches.find((m) => m.itemId === item.id);
    const fresh = match.candidates.some((c) => c.selectable && now - Date.parse(c.product.retrievedAt) <= 864e5);
    if (forceItemId === item.id) return [{ id: crypto.randomUUID(), itemId: item.id, query: searchQuery(item, online.profile), requestedAt: new Date(now).toISOString(), requestKey, retailers: ["instacart"], status: "queued", reason: "user-requested" }];
    if (item.savedProduct && previous?.reason !== "user-requested") return [];
    if (previous) {
      if (fresh && online.catalogRetrievedAt > previous.requestedAt && ["queued", "searching"].includes(previous.status)) return [{ ...previous, status: "ready-for-review", completedAt: online.catalogRetrievedAt }];
      return [previous];
    }
    if (fresh) return [];
    return [{ id: crypto.randomUUID(), itemId: item.id, query: searchQuery(item, online.profile), requestedAt: new Date(now).toISOString(), requestKey, retailers: ["instacart"], status: "queued", reason: match.candidates.length ? "stale-evidence" : "no-verified-match" }];
  });
}
function searchQuery(item, profile) {
  const choice = profile.confirmedChoices?.find((c) => norm(c.requestName) === norm(item.name) && (!item.preference.brand || norm(item.preference.brand) === norm(c.brand)) && (!item.preference.variant || norm(item.preference.variant) === norm(c.variant)) && (!item.preference.size || norm(item.preference.size) === norm(c.size)));
  return [...new Set([item.name, item.preference.brand || choice?.brand, item.preference.variant || choice?.variant, item.preference.size || choice?.size].filter(Boolean))].join(" ").slice(0, 180);
}

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/shopping/catalog.ts
var product = (id, name, brand, variant, size, unitQuantity, unit, image, attributes, allergens, prices, packCount = 1) => ({
  id,
  name,
  brand,
  variant,
  size,
  packCount,
  unitQuantity,
  unit,
  gtin: null,
  attributes,
  allergens,
  image: { url: `/grocery/${image}.svg`, kind: "demo-illustration", sourceUrl: null, permission: "local-asset" },
  offers: Object.entries(prices).map(([retailer, priceCents]) => ({ retailer, sku: `DEMO-${retailer}-${id}`, priceCents, productUrl: null }))
});
var CATALOG = [
  product("oatly-original", "Oatly Original Oatmilk", "Oatly", "Original", "64 fl oz", 64, "fl oz", "milk", ["dairy-free", "vegan"], ["oats"], { safeway: 549, instacart: 599, amazon: 649 }),
  product("oatly-fullfat", "Oatly Full Fat Oatmilk", "Oatly", "Full Fat", "64 fl oz", 64, "fl oz", "milk", ["dairy-free", "vegan"], ["oats"], { safeway: 549, instacart: 599 }),
  product("eggs-12", "Vital Farms Pasture-Raised Eggs", "Vital Farms", "Large \xB7 pasture-raised", "12 eggs", 12, "egg", "eggs", ["pasture-raised"], ["egg"], { safeway: 799, instacart: 849, amazon: 829 }),
  product("bananas", "Fresh Bananas", "Fresh produce", "Conventional", "1 lb bunch", 1, "lb", "bananas", ["vegan"], [], { safeway: 89, instacart: 99 }),
  product("bananas-organic", "Organic Bananas", "Fresh produce", "Organic", "1 lb bunch", 1, "lb", "bananas", ["vegan", "organic"], [], { safeway: 129, instacart: 139, amazon: 159 }),
  product("yogurt", "FAGE Total 5% Greek Yogurt", "FAGE", "Plain \xB7 whole milk", "32 oz", 32, "oz", "yogurt", [], ["milk"], { safeway: 699, instacart: 729, amazon: 649 }),
  product("berries", "Driscoll\u2019s Organic Blueberries", "Driscoll\u2019s", "Organic", "18 oz", 18, "oz", "berries", ["organic", "vegan"], [], { instacart: 1099, safeway: 1199 }),
  product("bread", "Dave\u2019s Killer Bread 21 Whole Grains", "Dave\u2019s Killer Bread", "21 Whole Grains and Seeds", "27 oz", 27, "oz", "bread", ["organic", "vegan"], ["wheat", "sesame"], { safeway: 679, instacart: 729, amazon: 699 }),
  product("oil", "Kirkland Signature Extra Virgin Olive Oil", "Kirkland Signature", "Extra virgin", "2 L", 2, "L", "oil", ["vegan"], [], { amazon: 2399 })
];

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/shopping/import.ts
var snapshotSchema = external_exports.object({
  version: external_exports.literal(1),
  ownerId: external_exports.string().min(1).max(240),
  sourceLabel: external_exports.string().min(1).max(160),
  retrievedAt: external_exports.string().datetime({ offset: true }),
  kitchen: external_exports.array(external_exports.record(external_exports.unknown())).max(5e3),
  list: external_exports.array(external_exports.record(external_exports.unknown())).max(1e3),
  receiptStores: external_exports.array(external_exports.object({ ownerId: external_exports.string().min(1).max(200), name: external_exports.string().min(1).max(180), receiptCount: external_exports.number().int().positive(), lastSeenAt: external_exports.string().nullable() }).strict()).max(40).optional(),
  recipes: external_exports.array(external_exports.object({ id: external_exports.string().min(1).max(200), ownerId: external_exports.string().min(1).max(200), title: external_exports.string().min(1).max(255), ingredients: external_exports.array(external_exports.string().min(1).max(2e3)).min(1).max(60), instructions: external_exports.array(external_exports.string().min(1).max(2e3)).min(1).max(60), imageUrl: external_exports.string().nullable(), category: external_exports.string().max(100), savedAt: external_exports.string().nullable() }).strict()).max(200).optional()
}).strict();
var knownText = (v) => {
  if (typeof v !== "string" && typeof v !== "number") return null;
  const s = String(v).trim();
  return !s || /^(?:unknown(?: brand| item)?|n\/a|null|undefined)$/i.test(s) ? null : s;
};
var textAt = (row, ...keys) => {
  for (const key of keys) {
    const value = knownText(row[key]);
    if (value !== null) return value;
  }
  return null;
};
var numberAt = (row, key) => {
  const v = row[key];
  if (v === null || v === void 0 || v === "" || typeof v === "boolean") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};
function sourceImage(value, allowedHosts) {
  if (!value) return null;
  try {
    const u = new URL(value);
    if (u.protocol !== "https:" || u.username || u.password || u.port || !allowedHosts.includes(u.hostname)) return null;
    if ([...u.searchParams.keys()].some((k) => /token|credential|signature|authorization|x-amz|x-goog|api.?key/i.test(k))) return null;
    u.hash = "";
    return u.toString();
  } catch {
    return null;
  }
}
function sourceEmoji(value) {
  if (!value.startsWith("emoji:")) return null;
  const emoji = value.slice(6).trim();
  if (!emoji || [...emoji].length > 24 || !new RegExp("^(?:\\p{Extended_Pictographic}|\\p{Regional_Indicator}|\\p{Emoji_Modifier}|\\uFE0F|\\u200D|\\u20E3|[0-9#*])+$", "u").test(emoji)) return null;
  if (!new RegExp("\\p{Extended_Pictographic}|\\p{Regional_Indicator}|[0-9#*]\\uFE0F?\\u20E3", "u").test(emoji)) return null;
  if ([...new Intl.Segmenter("en", { granularity: "grapheme" }).segment(emoji)].length > 3) return null;
  return emoji;
}
function sourceFor(row, kind, retrievedAt, hosts) {
  const originalId = textAt(row, "_id", "id", "itemUUID", "household_item_uuid");
  const originalName = ["product_name", "name", "title"].map((key) => typeof row[key] === "string" ? row[key].trim() : null).find((value) => value) ?? null;
  if (!originalId || !originalName || originalName.length > 160) throw new Error("Source item lacks a usable ID or product name.");
  const imageCandidates = ["resized_image_url", "product_image_url", "image_url", "image", "images"].map((key) => textAt(row, key)).filter((v) => v !== null);
  const imageUrl = imageCandidates.map((value) => sourceImage(value, hosts)).find((value) => value !== null) ?? null;
  const brand = textAt(row, "product_brand", "brand"), variant = textAt(row, "variant", "product_variant"), size = textAt(row, "product_size", "package_size", "size"), barcode = textAt(row, "product_barcode", "barcode");
  const uncertainFields = Object.entries({ brand, variant, size, barcode }).filter(([, value]) => value !== null && (value.length > 160 || /\b(likely|possibly|possible|unclear|uncertain|appears|may be|might be|guess)\b|\?/i.test(value))).map(([key]) => key);
  const rawPack = numberAt(row, "pack_count");
  const packCount = rawPack !== null && Number.isInteger(rawPack) && rawPack >= 1 && rawPack <= 100 ? rawPack : null;
  const quantityText = textAt(row, "quantity");
  const qty = quantityText !== null ? Number(quantityText) : NaN;
  return {
    kind,
    originalId,
    originalName,
    householdItemUUID: textAt(row, "itemUUID", "household_item_uuid"),
    retrievedAt,
    brand,
    variant,
    size,
    barcode,
    packCount,
    uncertainFields,
    // Kitchen display uses the app's emoji; retain the photo independently as source evidence.
    imageUrl,
    imageEmoji: imageCandidates.map(sourceEmoji).find((value) => value !== null) ?? null,
    imageWasPresent: imageCandidates.length > 0,
    quantityText,
    remainingQuantity: textAt(row, "remaining_quantity"),
    quantityValue: numberAt(row, "quantity_value"),
    quantityUnit: textAt(row, "quantity_unit"),
    fillPercent: numberAt(row, "fill_percent"),
    action: textAt(row, "action"),
    ...row.product_expiration !== void 0 ? { expiry: expiryEvidence(row.product_expiration, row.expiration_provenance ?? row.overrides__expiration_provenance, retrievedAt, typeof row._createdDate === "string" ? row._createdDate : null) } : {},
    quantityBasis: kind === "trepo-shopping-list" && Number.isInteger(qty) && qty >= 1 && qty <= 24 ? "source-list-count" : "local-restock-default"
  };
}
function preference(source) {
  return {
    brand: source.uncertainFields.includes("brand") ? "" : source.brand ?? "",
    variant: source.uncertainFields.includes("variant") ? "" : source.variant ?? "",
    size: source.uncertainFields.includes("size") ? "" : source.size ?? "",
    packCount: source.packCount,
    gtin: !source.uncertainFields.includes("barcode") && source.barcode && normalizeGTIN(source.barcode) ? source.barcode : null,
    productId: null,
    // Nutrition/ingredients are not inferred to be user restrictions or acceptable substitutions.
    requiredAttributes: [],
    excludedAllergens: [],
    substitutionProductIds: []
  };
}
function baseItem(row, source) {
  return {
    id: crypto.randomUUID(),
    name: source.originalName,
    category: textAt(row, "aisle_category", "category") ?? "Uncategorized",
    image: source.imageUrl ?? "",
    quantity: source.quantityBasis === "source-list-count" ? Number(source.quantityText) : 1,
    preference: preference(source),
    selectedProductId: null,
    sourceKitchenId: null,
    source
  };
}
function importTrepoSnapshot(input, scope) {
  const parsed = snapshotSchema.safeParse(input);
  if (!parsed.success) throw new Error("Invalid private Trepo import snapshot.");
  const s = parsed.data;
  if (s.ownerId !== scope.verifiedOwnerId || !scope.allowedRecordOwnerIds.includes(scope.verifiedOwnerId)) throw new Error("Snapshot owner does not match the independently verified account scope.");
  if (!/^trepo-copy-[a-f0-9-]{36}$/.test(scope.localHouseholdId)) throw new Error("Real data requires a distinct local-copy household ID.");
  const owners = new Set(scope.allowedRecordOwnerIds);
  for (const row of [...s.kitchen, ...s.list]) {
    for (const key of ["_owner", "ownerId", "owner_id"]) {
      const owner = knownText(row[key]);
      if (owner !== null && !owners.has(owner)) throw new Error("Source row is outside the verified owner scope.");
    }
  }
  const report = { kitchenCount: 0, listCount: 0, excludedInactiveKitchen: 0, excludedCompletedList: 0, omittedUnsafeImages: 0 };
  const kitchen = [], list2 = [], ids = /* @__PURE__ */ new Set();
  for (const [kind, rows] of [["trepo-kitchen", s.kitchen], ["trepo-shopping-list", s.list]]) {
    for (const row of rows) {
      const action = textAt(row, "action")?.toUpperCase();
      if (kind === "trepo-kitchen" && ["OUT", "DISCARDED", "ARCHIVED", "DELETED", "REMOVED"].includes(action ?? "")) {
        report.excludedInactiveKitchen++;
        continue;
      }
      if (kind === "trepo-shopping-list" && (row.checked === true || ["CHECKED", "REMOVED", "DELETED", "COMPLETED"].includes(action ?? ""))) {
        report.excludedCompletedList++;
        continue;
      }
      const source = sourceFor(row, kind, s.retrievedAt, scope.allowedImageHosts), key = `${kind}:${source.originalId}`;
      if (ids.has(key)) throw new Error("Duplicate source item ID; reconcile the snapshot before importing.");
      ids.add(key);
      if (source.imageWasPresent && !source.imageUrl && !source.imageEmoji) report.omittedUnsafeImages++;
      const item = baseItem(row, source);
      if (kind === "trepo-kitchen") {
        const measured = source.quantityValue !== null && source.quantityUnit ? `${source.quantityValue} ${source.quantityUnit}` : null;
        kitchen.push({
          ...item,
          remaining: source.remainingQuantity ?? measured ?? (source.fillPercent !== null ? `${source.fillPercent}%` : "Not recorded"),
          location: textAt(row, "storage_location") ?? "Not recorded",
          low: null
        });
      } else list2.push(item);
    }
  }
  report.kitchenCount = kitchen.length;
  report.listCount = list2.length;
  const savedRecipes = (s.recipes ?? []).map(({ ownerId, ...recipe }) => {
    if (!scope.allowedRecordOwnerIds.includes(ownerId) || !recipe.id.startsWith(ownerId + ":")) throw Error("Recipe source owner mismatch");
    return { ...recipe, imageUrl: sourceImage(recipe.imageUrl, scope.allowedImageHosts) };
  });
  if (new Set(savedRecipes.map((r) => r.id)).size !== savedRecipes.length) throw Error("Duplicate recipe source");
  const receiptStores = (s.receiptStores ?? []).map(({ ownerId, ...store }) => {
    if (!owners.has(ownerId)) throw Error("Receipt store owner mismatch");
    return store;
  });
  return { household: {
    savedRecipes,
    receiptStores,
    id: scope.localHouseholdId,
    name: scope.localHouseholdName,
    revision: 0,
    purchaseRevision: 0,
    provenance: { kind: "trepo-readonly-copy", sourceLabel: s.sourceLabel, retrievedAt: s.retrievedAt, localEditsOnly: true },
    kitchen,
    list: list2,
    savedPreferences: Object.fromEntries(kitchen.map((k) => [k.id, structuredClone(k.preference)])),
    config: { budgetCents: 1e4, tipCentsPerCart: 400, strategy: "single", retailer: "safeway", deliveryWindow: "tomorrow-am", zip: null, addressLabel: null, householdMode: "unset" },
    quotes: [],
    orders: []
  }, report };
}

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/shopping/persistence.ts
function serializeHousehold(h) {
  const { kitchenKnowledge: _knowledge, ...draft } = h;
  return JSON.stringify(draft);
}

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/shopping/session-import.ts
var bundleSchema = external_exports.object({ snapshot: external_exports.unknown(), history: external_exports.unknown(), knowledge: external_exports.unknown(), allowedRecordOwnerIds: external_exports.array(external_exports.string().regex(/^(?:[a-f0-9-]{36}|\d{4,6})$/)).min(1).max(50), allowedImageHosts: external_exports.array(external_exports.string().max(200)).max(50) }).strict();
async function prepareSessionDraft(actor, input, now = Date.now()) {
  if (!/^trepo:[a-f0-9-]{36}$/.test(actor)) throw new Error("Invalid session actor");
  const userId = actor.slice(6), bundle = bundleSchema.parse(input), id = `trepo-copy-${crypto.randomUUID()}`;
  const { household: original } = importTrepoSnapshot(bundle.snapshot, { verifiedOwnerId: userId, allowedRecordOwnerIds: bundle.allowedRecordOwnerIds, allowedImageHosts: bundle.allowedImageHosts, localHouseholdId: id, localHouseholdName: "Your kitchen" });
  const age = now - Date.parse(original.provenance.retrievedAt);
  if (age < -6e4 || age > 3e5) throw new Error("Snapshot is not fresh");
  const { profile } = await buildUserProfile(bundle.history, userId, new Date(now).toISOString());
  const online = attachOnlineShopping(original, profile, [], original.provenance.retrievedAt, now);
  const { household } = applyKitchenKnowledge(online, bundle.knowledge, userId, now);
  const serialized = serializeHousehold(household), knowledge = JSON.stringify(household.kitchenKnowledge);
  if (new TextEncoder().encode(serialized).length > 18e5 || new TextEncoder().encode(knowledge).length > 18e5) throw new Error("Snapshot exceeds local storage limits");
  return { household, serialized, knowledge };
}

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/shopping/shopping-profile.ts
var DAY3 = 864e5;
var VERSION = 1;
var norm2 = (value) => value.normalize("NFKC").toLocaleLowerCase().replace(/[^\p{L}\p{N}%]+/gu, " ").trim();
var clean2 = (value, max = 160) => value?.trim().replace(/\s+/g, " ").slice(0, max) || null;
var week = (value) => {
  const date2 = new Date(value);
  date2.setUTCDate(date2.getUTCDate() - (date2.getUTCDay() + 6) % 7);
  return date2.toISOString().slice(0, 10);
};
var distinct = (values) => [...new Set(values)];
var sorted = (values) => values.sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
var fingerprintPreference = (p) => [p.brand, p.variant, p.size, p.packCount, p.gtin, p.requiredAttributes, p.excludedAllergens];
async function shoppingProfileFingerprint(h) {
  const history = h.onlineShopping?.profile, knowledge = h.kitchenKnowledge;
  const data = JSON.stringify([
    VERSION,
    "diverse-category-2026-09-24",
    h.id,
    history?.subjectActorId ?? null,
    sorted(h.kitchen.map((k) => [k.id, k.name, k.category, fingerprintPreference(h.savedPreferences[k.id] ?? k.preference), k.source ? [k.source.originalId, k.source.uncertainFields, k.source.retrievedAt] : null])),
    sorted((history?.observations ?? []).map((o) => [o.id, o.subjectActorId, o.sourceRecordId, o.attribution, o.confidence, o.observedAt, o.name, o.brand, o.variant, o.size, o.category, observationActivity(o).kind])),
    knowledge ? [knowledge.subjectActorId, sorted(knowledge.events.map((e) => [e.id, e.actorId, e.source, e.sourceId, e.operationId, e.itemSourceId, e.occurredAt, e.kind, e.sourceAction, e.name, e.brand, e.undoneBy]))] : null,
    sorted((h.shoppingRequestPreferences ?? []).map((p) => [p.original, p.preferred, p.source, p.confirmedAt]))
  ]);
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(data)))].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
var count = external_exports.number().int().nonnegative();
var evidenceSchema2 = external_exports.object({ observations: count, currentItems: count, distinctDays: count, distinctWeeks: count, lastSeenAt: external_exports.string().nullable(), explicitConsumedDays: count, explicitPurchasedDays: count, explicitDiscardedDays: count, selectedDays: count.optional() }).strict();
var productSchema2 = external_exports.object({ id: external_exports.string(), name: external_exports.string(), brand: external_exports.string().nullable(), variant: external_exports.string().nullable(), size: external_exports.string().nullable(), category: external_exports.string().nullable(), confidence: external_exports.enum(["low", "medium"]), eligible: external_exports.boolean(), evidence: evidenceSchema2 }).strict();
var patternSchema = external_exports.object({ id: external_exports.string(), kind: external_exports.enum(["brand", "product-label"]), value: external_exports.string(), productCount: count, observations: count, distinctDays: count, distinctWeeks: count, lastSeenAt: external_exports.string().nullable(), confidence: external_exports.enum(["low", "medium"]) }).strict();
var shoppingPreferenceProfileSchema = external_exports.object({ version: external_exports.literal(VERSION), householdId: external_exports.string(), fingerprint: external_exports.string(), generatedAt: external_exports.string(), status: external_exports.literal("observed-shopping-context"), products: external_exports.array(productSchema2).max(192), patterns: external_exports.array(patternSchema).max(20), explicitRequests: external_exports.array(external_exports.object({ original: external_exports.string(), preferred: external_exports.string(), confirmedAt: external_exports.string() }).strict()).max(100), coverage: external_exports.object({ historyDays: count, currentItems: count, personalObservations: count, personalEvents: count, completeHistory: external_exports.literal(false) }).strict(), caveat: external_exports.string() }).strict();
var labels = [["organic", /(?<!non[- ])(?<!not )(?<!no )\borganic\b/i], ["non-GMO", /\bnon[ -]?gmo\b/i], ["unsweetened", /\bunsweetened\b/i], ["whole grain", /\bwhole[ -]grains?\b/i], ["grass-fed", /\bgrass[ -]fed\b/i], ["pasture-raised", /\bpasture[ -]raised\b/i]];
async function buildShoppingPreferenceProfile(h, now = Date.now(), fingerprint2) {
  const history = h.onlineShopping?.profile, subject = history?.subjectActorId ?? h.kitchenKnowledge?.subjectActorId;
  const validDate2 = (value) => !!value && Number.isFinite(Date.parse(value)) && Date.parse(value) <= now && Date.parse(value) >= now - 730 * DAY3;
  const groups = /* @__PURE__ */ new Map(), sourceGroups = /* @__PURE__ */ new Map();
  const group = (name, brand, variant, size, category) => {
    const key = JSON.stringify([norm2(name), norm2(brand ?? ""), norm2(variant ?? ""), norm2(size ?? "")]);
    if (!groups.has(key)) groups.set(key, { name: clean2(name, 240), brand: clean2(brand, 80), variant: clean2(variant, 100), size: clean2(size, 80), category: clean2(category, 80), records: /* @__PURE__ */ new Map(), current: /* @__PURE__ */ new Set(), dates: /* @__PURE__ */ new Set(), used: /* @__PURE__ */ new Set(), bought: /* @__PURE__ */ new Set(), discarded: /* @__PURE__ */ new Set(), labels: /* @__PURE__ */ new Set() });
    return groups.get(key);
  };
  for (const item of h.kitchen) {
    const p = h.savedPreferences[item.id] ?? item.preference, uncertain2 = new Set(item.source?.uncertainFields ?? []);
    const g = group(item.name, uncertain2.has("brand") ? null : p.brand || null, uncertain2.has("variant") ? null : p.variant || null, uncertain2.has("size") ? null : p.size || null, item.category || null);
    const key = item.source?.originalId ?? `current:${item.id}`;
    g.current.add(key);
    g.records.set(key, null);
    sourceGroups.set(key, g);
    const labelText = [item.name, p.variant, ...p.requiredAttributes].join(" ");
    for (const [label, pattern] of labels) if (pattern.test(labelText)) g.labels.add(label);
  }
  const seen = /* @__PURE__ */ new Set();
  let personalObservations = 0;
  for (const o of history?.observations ?? []) {
    if (!subject || o.subjectActorId !== subject || o.attribution !== "recorded-actor" || o.confidence !== "recorded" || !validDate2(o.observedAt) || seen.has(o.sourceRecordId)) continue;
    if (!["inventory-in", "explicit-purchased", "explicit-consumed"].includes(observationActivity(o).kind)) continue;
    seen.add(o.sourceRecordId);
    personalObservations++;
    const g = sourceGroups.get(o.sourceRecordId) ?? group(o.name, o.brand, o.variant, o.size, o.category), date2 = new Date(o.observedAt).toISOString().slice(0, 10);
    sourceGroups.set(o.sourceRecordId, g);
    g.records.set(o.sourceRecordId, date2);
    g.dates.add(date2);
    for (const [label, pattern] of labels) if (pattern.test([o.name, o.variant].join(" "))) g.labels.add(label);
  }
  const events = subject && h.kitchenKnowledge?.subjectActorId === subject ? dedupeKitchenEvents(h.kitchenKnowledge.events, subject).events.filter((e) => validDate2(e.occurredAt) && e.name?.trim()) : [];
  for (const e of events) {
    if (!["consumed", "purchased", "inventory-in", "discarded"].includes(e.kind)) continue;
    const g = (e.itemSourceId ? sourceGroups.get(e.itemSourceId) : void 0) ?? [...groups.values()].find((g2) => norm2(g2.name) === norm2(e.name) && norm2(g2.brand ?? "") === norm2(e.brand ?? "")) ?? group(e.name, e.brand, null, null, null), date2 = new Date(e.occurredAt).toISOString().slice(0, 10);
    if (e.kind === "discarded") g.discarded.add(date2);
    else {
      g.dates.add(date2);
      if (e.kind === "consumed") g.used.add(date2);
      if (e.kind === "purchased") g.bought.add(date2);
      if (e.kind === "inventory-in") g.records.set(e.itemSourceId ?? `event:${e.id}`, date2);
    }
    for (const [label, pattern] of labels) if (pattern.test(e.name)) g.labels.add(label);
  }
  const ranked = [...groups.entries()].map(([key, g]) => {
    const days2 = g.dates.size, weeks = new Set([...g.dates].map(week)).size;
    const eligible = weeks >= 2 && days2 >= 2 && (g.discarded.size < 2 || g.discarded.size < g.used.size);
    const lastSeenAt = [...g.dates].sort().at(-1) ?? null;
    return { key, g, eligible, lastSeenAt, score: (eligible ? 100 : 0) + (g.current.size ? 25 : 0) + Math.min(20, weeks * 3) + (lastSeenAt ? Math.max(0, 20 - (now - Date.parse(lastSeenAt)) / DAY3 / 36) : 0) };
  }).sort((a, b) => b.score - a.score || a.key.localeCompare(b.key));
  const products = ranked.slice(0, 192).map(({ g, eligible, lastSeenAt }, i) => ({ id: `product-${i + 1}`, name: g.name, brand: g.brand, variant: g.variant, size: g.size, category: g.category, confidence: eligible ? "medium" : "low", eligible, evidence: { observations: g.records.size, currentItems: g.current.size, distinctDays: g.dates.size, distinctWeeks: new Set([...g.dates].map(week)).size, lastSeenAt, explicitConsumedDays: g.used.size, explicitPurchasedDays: g.bought.size, explicitDiscardedDays: g.discarded.size } }));
  const patterns = /* @__PURE__ */ new Map();
  for (const { g } of ranked) {
    if (g.discarded.size >= 2 && g.discarded.size >= g.used.size) continue;
    const values = [...g.brand ? [["brand", g.brand]] : [], ...[...g.labels].map((label) => ["product-label", label])];
    for (const [kind, value] of values) {
      const key = `${kind}:${norm2(value)}`;
      if (!patterns.has(key)) patterns.set(key, { kind, value, groups: [] });
      patterns.get(key).groups.push(g);
    }
  }
  const recurring = [...patterns.values()].map((pattern) => {
    const dates = distinct(pattern.groups.flatMap((g) => [...g.dates])), observations = pattern.groups.reduce((sum, g) => sum + g.records.size, 0), current = pattern.groups.reduce((sum, g) => sum + g.current.size, 0), weeks = distinct(dates.map(week)).length;
    return { ...pattern, observations, current, dates, weeks };
  }).filter((p) => p.groups.length >= 3 && (p.weeks >= 2 || p.current >= 3)).sort((a, b) => b.groups.length - a.groups.length || b.weeks - a.weeks || a.value.localeCompare(b.value)).slice(0, 20).map((p, i) => ({ id: `pattern-${i + 1}`, kind: p.kind, value: p.value, productCount: p.groups.length, observations: p.observations, distinctDays: p.dates.length, distinctWeeks: p.weeks, lastSeenAt: p.dates.sort().at(-1) ?? null, confidence: p.weeks >= 2 ? "medium" : "low" }));
  return { version: VERSION, householdId: h.id, fingerprint: fingerprint2 ?? await shoppingProfileFingerprint(h), generatedAt: new Date(now).toISOString(), status: "observed-shopping-context", products, patterns: recurring, explicitRequests: (h.shoppingRequestPreferences ?? []).filter((p) => p.source === "user-stated").slice(-100).map((p) => ({ original: p.original.slice(0, 160), preferred: p.preferred.slice(0, 160), confirmedAt: p.confirmedAt })), coverage: { historyDays: 730, currentItems: h.kitchen.length, personalObservations, personalEvents: events.length, completeHistory: false }, caveat: "Kitchen observations are not purchases or confirmed preferences. Product labels do not establish a diet, allergy, health condition or religious belief. Explicit requests always win." };
}
var specificitySchema = external_exports.object({ category: external_exports.string().trim().min(1).max(120), explicitDetails: external_exports.array(external_exports.string().trim().min(1).max(120)).max(12), specificity: external_exports.enum(["broad", "specific"]) }).strict();
var planSchema = external_exports.object({ productId: external_exports.string().nullable().optional(), productIds: external_exports.array(external_exports.string()).max(3).optional(), patternIds: external_exports.array(external_exports.string()).max(2) }).strict();

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/shopping/restock-suggestions.ts
var DAY4 = 864e5;
var norm3 = (value) => value.normalize("NFKC").toLowerCase().replace(/\b(oat|almond|soy)milk\b/g, "$1 milk").replace(/[^\p{L}\p{N}]+/gu, " ").trim();
var words2 = (value) => norm3(value).split(" ").filter(Boolean).map((word) => word.length > 4 && word.endsWith("s") ? word.slice(0, -1) : word);
var contains = (value, phrase) => {
  const parts = words2(phrase), target = words2(value);
  return parts.length > 0 && parts.every((part) => target.includes(part));
};
var identity = (name, brand) => {
  const n = norm3(name), b = norm3(brand ?? "");
  return b && !contains(n, b) ? `${b} ${n}` : n;
};
var week2 = (date2) => {
  const d = new Date(date2);
  d.setUTCDate(d.getUTCDate() - (d.getUTCDay() + 6) % 7);
  return d.toISOString().slice(0, 10);
};
var last = (dates) => [...dates].sort().at(-1) ?? null;
var low = (item) => item.low === true || item.source?.fillPercent != null && item.source.fillPercent >= 0 && item.source.fillPercent <= 25;
var itemNames = (item) => [item.name, item.source?.originalName, item.savedProduct?.name].filter((value) => !!value);
function restockSuggestions(h, options = {}, now = Date.now()) {
  const limit = options.limit ?? 5, days2 = options.historyDays ?? 180;
  if (!Number.isInteger(limit) || limit < 1 || limit > 10 || !Number.isInteger(days2) || days2 < 7 || days2 > 730) throw new Error("Restock suggestion limits are outside the supported range.");
  const start = now - days2 * DAY4, profile = h.onlineShopping?.profile, subject = profile?.subjectActorId ?? h.kitchenKnowledge?.subjectActorId;
  const validDate2 = (date2) => !!date2 && Number.isFinite(Date.parse(date2)) && Date.parse(date2) >= start && Date.parse(date2) <= now;
  const groups = /* @__PURE__ */ new Map();
  const group = (name, brand, sourceId = null) => {
    const kitchen = h.kitchen.filter((item) => sourceId && item.source?.originalId === sourceId || identity(item.name, item.source?.brand ?? item.preference.brand) === identity(name, brand));
    const first = kitchen[0], key = identity(first?.name ?? name, first?.source?.brand ?? first?.preference.brand ?? brand);
    if (!groups.has(key)) groups.set(key, { key, name: first?.name ?? name, brand: first?.source?.brand || first?.preference.brand || brand, variant: first?.source?.variant || first?.preference.variant || null, size: first?.source?.size || first?.preference.size || null, kitchen, used: /* @__PURE__ */ new Set(), bought: /* @__PURE__ */ new Set(), added: /* @__PURE__ */ new Set(), observed: /* @__PURE__ */ new Map(), discarded: /* @__PURE__ */ new Set() });
    return groups.get(key);
  };
  for (const item of h.kitchen) group(item.name, item.source?.brand ?? item.preference.brand, item.source?.originalId ?? null);
  const knowledge = h.kitchenKnowledge?.subjectActorId === subject ? h.kitchenKnowledge : null;
  const events = subject ? dedupeKitchenEvents(knowledge?.events ?? [], subject).events.filter((event) => validDate2(event.occurredAt)) : [];
  for (const event of events) {
    if (!event.name?.trim()) continue;
    const g = group(event.name, event.brand, event.itemSourceId), date2 = new Date(event.occurredAt).toISOString().slice(0, 10);
    if (event.kind === "consumed") g.used.add(date2);
    else if (event.kind === "purchased") g.bought.add(date2);
    else if (event.kind === "inventory-in") g.added.add(date2);
    else if (event.kind === "discarded") g.discarded.add(date2);
  }
  const seen = /* @__PURE__ */ new Set();
  const observations = (profile?.observations ?? []).filter((observation) => {
    if (!subject || observation.subjectActorId !== subject || observation.attribution !== "recorded-actor" || !validDate2(observation.observedAt) || seen.has(observation.id)) return false;
    seen.add(observation.id);
    return true;
  });
  for (const observation of observations) {
    if (observation.confidence !== "recorded" || observationActivity(observation).kind !== "inventory-in") continue;
    const g = group(observation.name, observation.brand, observation.sourceRecordId);
    g.variant ??= observation.variant;
    g.size ??= observation.size;
    if (!g.observed.has(observation.sourceRecordId)) g.observed.set(observation.sourceRecordId, new Date(observation.observedAt).toISOString().slice(0, 10));
  }
  const excluded = new Set(options.excludeIds ?? []), ranked = [], usedQueries = /* @__PURE__ */ new Set();
  for (const g of groups.values()) {
    const id = `restock:${g.key}`;
    if (excluded.has(id)) continue;
    if (g.discarded.size >= 2 && g.discarded.size >= g.used.size) continue;
    const observedDates = new Set(g.observed.values()), repeatDates = g.added.size ? g.added : observedDates;
    const repeated = repeatDates.size >= 2 && new Set([...repeatDates].map(week2)).size >= 2;
    const currentLow = g.kitchen.length > 0 && g.kitchen.every(low), stock = g.kitchen.length ? currentLow ? "low" : "present" : "not-in-snapshot";
    const basis = currentLow ? "low-stock" : g.used.size ? "recorded-usage" : g.bought.size >= 2 ? "repeat-purchases" : repeated ? "repeat-additions" : null;
    if (!basis) continue;
    let name = g.name, query = [g.brand && !contains(g.name, g.brand) ? g.brand : null, g.name, g.variant && !contains(g.name, g.variant) ? g.variant : null, g.size && !contains(g.name, g.size) ? g.size : null].filter(Boolean).join(" ").slice(0, 160);
    const preference2 = [...h.shoppingRequestPreferences ?? []].sort((a, b) => b.original.length - a.original.length).find((value) => contains(query, value.original));
    if (preference2) {
      name = preference2.preferred;
      query = preference2.preferred;
    }
    if (h.list.some((item2) => g.kitchen.some((k) => item2.sourceKitchenId === k.id) || itemNames(item2).some((value) => contains(query, value) || contains(g.name, value)))) continue;
    const dates = basis === "recorded-usage" ? g.used : basis === "repeat-purchases" ? g.bought : repeatDates;
    const distinctWeeks = new Set([...dates].map(week2)).size;
    const latest = last(dates);
    const usageDate = latest ? new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" }).format(/* @__PURE__ */ new Date(`${latest}T12:00:00Z`)) : null;
    let reason = basis === "low-stock" ? "Running low in your saved kitchen" : basis === "recorded-usage" ? distinctWeeks > 1 ? `You logged using this in ${distinctWeeks} different weeks` : `You logged using this ${dates.size === 1 ? `on ${usageDate}` : `on ${dates.size} different days`}` : basis === "repeat-purchases" ? `You bought this on ${dates.size} different days` : g.added.size ? `You added this in ${distinctWeeks} different weeks` : `Shows up in your kitchen across ${distinctWeeks} weeks`;
    if (stock === "present" && basis !== "low-stock") reason += " \xB7 check what\u2019s left";
    else if (stock === "not-in-snapshot") reason += " \xB7 not in your saved kitchen";
    const item = { id, name, brand: g.brand || null, query, reason, kitchenId: g.kitchen[0]?.id ?? null, emoji: g.kitchen[0]?.source?.imageEmoji ?? null, stock, evidence: { basis, distinctDays: dates.size, distinctWeeks, lastRecordedAt: last(dates) } };
    const recency = latest ? Math.max(0, 30 - (now - Date.parse(latest)) / DAY4 / (days2 / 30)) : 0;
    const score = (basis === "low-stock" ? 120 : basis === "recorded-usage" ? 80 : basis === "repeat-purchases" ? 65 : 40) + (stock === "not-in-snapshot" ? 15 : 0) + Math.min(20, distinctWeeks * 3) + recency;
    ranked.push({ item, score });
  }
  const candidates = ranked.sort((a, b) => b.score - a.score || a.item.id.localeCompare(b.item.id)).filter(({ item }) => {
    const key = norm3(item.query);
    if (usedQueries.has(key)) return false;
    usedQueries.add(key);
    return true;
  });
  const items = candidates.slice(0, limit).map(({ item }) => item);
  const period = days2 === 180 ? "last six months" : `last ${days2} days`;
  const summary = items.length ? `From your ${period} of kitchen history, here ${items.length === 1 ? "is one item" : "are " + items.length + " items"} worth checking. I\u2019ve left out what\u2019s already on your list.` : excluded.size ? "That\u2019s all I can support from your kitchen history. Your existing list is already covered." : "I don\u2019t have enough restock evidence yet. Your kitchen and shopping list are a good place to start.";
  return { asOf: new Date(now).toISOString(), items, requestedCount: limit, hasMore: candidates.length > limit, summary, coverage: { historyDays: days2, personalEvents: events.length, personalObservations: observations.length, retrievedAt: knowledge?.retrievedAt ?? profile?.retrievedAt ?? h.provenance?.retrievedAt ?? null, completeHistory: false } };
}

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/household-brain/core.ts
var DAY5 = 864e5;
var brainKey = (name, brand = "") => {
  const norm4 = (s) => s.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  const n = norm4(name), b = norm4(brand ?? "");
  return b && !n.includes(b) ? `${b} ${n}` : n;
};
var median = (a) => {
  const s = [...a].sort((a2, b) => a2 - b);
  return s.length ? (s[Math.floor((s.length - 1) / 2)] + s[Math.ceil((s.length - 1) / 2)]) / 2 : 0;
};
var days = (dates) => [...new Set(dates.map((d) => new Date(d).toISOString().slice(0, 10)))].sort();
var weekCount = (dates) => new Set(dates.map((value) => {
  const d = new Date(value);
  d.setUTCDate(d.getUTCDate() - (d.getUTCDay() + 6) % 7);
  return d.toISOString().slice(0, 10);
})).size;
async function brainFingerprint(h, policy, now = Date.now(), viewer = null) {
  const text3 = JSON.stringify(["household-brain-rules-2026-09-24-v4", h, policy, viewer, Math.floor(now / (15 * 6e4))]);
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text3)))].map((b) => b.toString(16).padStart(2, "0")).join("");
}
var learnedAt = (at, policy, now) => !!at && Number.isFinite(Date.parse(at)) && Date.parse(at) >= policy.learningSince && Date.parse(at) <= now && !(policy.pauses ?? []).some((p) => Date.parse(at) >= p.start && (p.end === null || Date.parse(at) < p.end));
function learningView(h, policy, now) {
  const view = structuredClone(h), suppressed = new Set(policy.suppressed);
  const allowed = (name, brand) => !suppressed.has(brainKey(name, brand));
  const after = (at) => learnedAt(at, policy, now);
  view.kitchen = view.kitchen.filter((i) => allowed(i.name, i.source?.brand ?? i.preference.brand));
  if (view.kitchenKnowledge) view.kitchenKnowledge.events = view.kitchenKnowledge.events.filter((e) => e.name && allowed(e.name, e.brand) && after(e.occurredAt));
  if (view.onlineShopping) view.onlineShopping.profile.observations = view.onlineShopping.profile.observations.filter((o) => allowed(o.name, o.brand) && after(o.observedAt));
  return view;
}
async function buildHouseholdBrain(h, activities, policy, now = Date.now(), fingerprint2) {
  const view = learningView(h, policy, now), profileView = structuredClone(view);
  if (policy.learningSince > 0 || policy.pauses?.length) profileView.kitchen = profileView.kitchen.filter((k) => learnedAt(k.source?.retrievedAt, policy, now));
  const profile = await buildShoppingPreferenceProfile(profileView, now, await shoppingProfileFingerprint(h));
  const subject = view.onlineShopping?.profile.subjectActorId ?? view.kitchenKnowledge?.subjectActorId;
  const events = subject && view.kitchenKnowledge?.subjectActorId === subject ? dedupeKitchenEvents(view.kitchenKnowledge.events, subject).events.filter((e) => e.name && e.occurredAt && Date.parse(e.occurredAt) <= now && Date.parse(e.occurredAt) >= now - 730 * DAY5) : [];
  const groups = /* @__PURE__ */ new Map();
  for (const e of events) {
    const key = brainKey(e.name, e.brand);
    groups.set(key, [...groups.get(key) ?? [], e]);
  }
  const buyingPatterns = [...groups.entries()].map(([id, records]) => {
    const bought = days(records.filter((e) => e.kind === "purchased").map((e) => e.occurredAt)), used = days(records.filter((e) => e.kind === "consumed").map((e) => e.occurredAt)), discarded = days(records.filter((e) => e.kind === "discarded").map((e) => e.occurredAt));
    const gaps = bought.slice(1).map((d, i) => (Date.parse(d) - Date.parse(bought[i])) / DAY5), middle = median(gaps);
    const credible = bought.length >= 3 && middle > 0 && Math.max(...gaps) <= middle * 2 && now - Date.parse(bought.at(-1)) <= Math.max(30, middle * 3) * DAY5;
    const from = credible ? Date.parse(bought.at(-1)) : 0;
    return {
      id,
      name: records[0].name,
      brand: records[0].brand,
      evidenceIds: records.map((e) => e.id),
      purchaseDays: bought.length,
      consumedDays: used.length,
      discardedDays: discarded.length,
      cadence: credible ? { medianDays: middle, minDays: Math.min(...gaps), maxDays: Math.max(...gaps), nextWindowStart: new Date(from + Math.min(...gaps) * DAY5).toISOString(), nextWindowEnd: new Date(from + Math.max(...gaps) * DAY5).toISOString(), confidence: bought.length >= 5 ? "medium" : "low" } : null
    };
  }).sort((a, b) => b.purchaseDays + b.consumedDays - a.purchaseDays - a.consumedDays || a.id.localeCompare(b.id));
  const valid = activities.filter((e) => learnedAt(e.occurredAt, policy, now) && Date.parse(e.receivedAt) <= now && !policy.suppressed.includes(brainKey(e.name, e.brand)));
  const choices = /* @__PURE__ */ new Map();
  for (const event of valid) if (event.kind === "product-selected") {
    const key = brainKey(event.name, event.brand);
    choices.set(key, [...choices.get(key) ?? [], event]);
  }
  for (const [key, records] of choices) {
    const selected = days(records.map((e) => e.occurredAt));
    const span = selected.length ? (Date.parse(selected.at(-1)) - Date.parse(selected[0])) / DAY5 : 0;
    if (selected.length < 2 || span < 7) continue;
    const wasted = buyingPatterns.find((p) => p.id === key);
    if (wasted && wasted.discardedDays >= 2 && wasted.discardedDays >= wasted.consumedDays) continue;
    const existing = profile.products.find((p) => brainKey(p.name, p.brand) === key);
    if (existing) {
      existing.evidence.selectedDays = selected.length;
      existing.eligible = true;
    } else profile.products.unshift({ id: `choice:${key}`, name: records[0].name, brand: records[0].brand, variant: null, size: null, category: null, confidence: "medium", eligible: true, evidence: { observations: 0, currentItems: 0, distinctDays: selected.length, distinctWeeks: weekCount(selected), lastSeenAt: selected.at(-1), explicitConsumedDays: 0, explicitPurchasedDays: 0, explicitDiscardedDays: 0, selectedDays: selected.length } });
  }
  profile.products = profile.products.slice(0, 192);
  const recipeMap = /* @__PURE__ */ new Map(), weekdays = {}, durations = [];
  const cooked = /* @__PURE__ */ new Set();
  for (const event of valid) {
    if (event.kind === "recipe-cooked") {
      const occasion = `${event.entityId}:${event.occurredAt.slice(0, 10)}`;
      if (cooked.has(occasion)) continue;
      cooked.add(occasion);
      const day2 = new Intl.DateTimeFormat("en-US", { weekday: "long", timeZone: "UTC" }).format(new Date(event.occurredAt));
      weekdays[day2] = (weekdays[day2] ?? 0) + 1;
      if (event.minutes) durations.push(event.minutes);
    }
    if (!["recipe-cooked", "recipe-saved", "recipe-rated"].includes(event.kind)) continue;
    for (const tag of new Set(event.tags.map((t) => brainKey(t)))) {
      const p = recipeMap.get(tag) ?? { tag, saved: 0, cooked: 0, liked: 0, disliked: 0, lastAt: event.occurredAt };
      if (event.kind === "recipe-saved") p.saved++;
      if (event.kind === "recipe-cooked") p.cooked++;
      if (event.kind === "recipe-rated" && event.rating) p[event.rating]++;
      if (event.occurredAt > p.lastAt) p.lastAt = event.occurredAt;
      recipeMap.set(tag, p);
    }
  }
  let pool = [];
  let base = restockSuggestions(view, { limit: 10, historyDays: 730 }, now);
  for (let page = 0; page < 5; page++) {
    const batch = page ? restockSuggestions(view, { limit: 10, historyDays: 730, excludeIds: pool.map((i) => i.id) }, now) : base;
    pool.push(...batch.items);
    if (!batch.hasMore) break;
  }
  for (const [key, records] of choices) {
    const dates = days(records.map((e) => e.occurredAt)), span = dates.length ? (Date.parse(dates.at(-1)) - Date.parse(dates[0])) / DAY5 : 0;
    const last2 = dates.at(-1);
    if (dates.length < 2 || span < 7 || !last2 || now - Date.parse(last2) > 60 * DAY5) continue;
    const wasted = buyingPatterns.find((p) => p.id === key);
    if (wasted && wasted.discardedDays >= 2 && wasted.discardedDays >= wasted.consumedDays) continue;
    if (pool.some((i) => brainKey(i.name, i.brand) === key) || h.list.some((i) => brainKey(i.savedProduct?.name ?? i.name, i.savedProduct?.brand ?? i.preference.brand) === key)) continue;
    pool.push({ id: `choice:${key}`, name: records[0].name, brand: records[0].brand, query: records[0].name.slice(0, 160), reason: `Chosen for your list on ${dates.length} different days \xB7 check what\u2019s left`, kitchenId: null, emoji: null, stock: "not-in-snapshot", evidence: { basis: "repeat-choices", distinctDays: dates.length, distinctWeeks: weekCount(dates), lastRecordedAt: last2 } });
  }
  const removed = new Set(valid.filter((a) => a.kind === "list-removed" && now - Date.parse(a.occurredAt) < 7 * DAY5).map((a) => brainKey(a.name, a.brand)));
  pool = pool.filter((i) => !removed.has(brainKey(i.name, i.brand)));
  const dismissed = new Set(valid.filter((a) => a.kind === "recommendation-dismissed" && now - Date.parse(a.occurredAt) < 7 * DAY5).map((a) => a.entityId));
  pool = pool.filter((item) => !dismissed.has(item.id));
  const scored = pool.map((item, index) => {
    const pattern = buyingPatterns.find((p) => p.id === brainKey(item.name, item.brand));
    const due = pattern?.cadence && Date.parse(pattern.cadence.nextWindowStart) <= now + 7 * DAY5;
    const reason = due ? `Recorded purchases were about ${Math.round(pattern.cadence.medianDays)} days apart \xB7 check what\u2019s left` : item.reason;
    return { item: { ...item, reason }, score: (item.stock === "low" ? 200 : 0) + (due ? 100 : 0) + 50 - index };
  }).sort((a, b) => b.score - a.score);
  base = { ...base, items: scored.map((x) => x.item), hasMore: scored.length > 5 };
  const shelf = queryShelfLife(view, now), at = h.provenance?.retrievedAt ?? h.kitchenKnowledge?.retrievedAt ?? h.onlineShopping?.profile.retrievedAt ?? null;
  const age = at && Number.isFinite(Date.parse(at)) ? Math.max(0, Math.floor((now - Date.parse(at)) / DAY5)) : null;
  const latestActivityAt = valid.map((e) => e.occurredAt).sort().at(-1) ?? null;
  return {
    version: 1,
    householdId: h.id,
    generation: policy.generation,
    revision: policy.revision,
    fingerprint: fingerprint2 ?? await brainFingerprint(h, policy, now),
    generatedAt: new Date(now).toISOString(),
    expiresAt: new Date((Math.floor(now / (15 * 6e4)) + 1) * 15 * 6e4).toISOString(),
    enabled: policy.enabled,
    profile,
    buyingPatterns,
    recipePatterns: [...recipeMap.values()].sort((a, b) => b.cooked + b.liked - a.cooked - a.liked).slice(0, 40),
    cooking: { recordedOccasions: cooked.size, typicalMinutes: durations.length >= 3 ? median(durations) : null, weekdayCounts: weekdays, interpretation: "Recorded cooking occasions in UTC; not confirmed meals eaten by an individual." },
    groceries: base,
    useSoon: shelf.items.filter((i) => i.daysUntil >= 0).map((i) => {
      const k = h.kitchen.find((k2) => k2.id === i.kitchenId);
      return { kitchenId: k.id, name: k.name, brand: k.source?.brand ?? k.preference.brand ?? null, daysUntil: i.daysUntil, date: i.deadline, basis: i.basis };
    }),
    checkDates: shelf.items.filter((i) => i.daysUntil < 0).map((i) => ({ kitchenId: i.kitchenId, name: h.kitchen.find((k) => k.id === i.kitchenId).name, daysPast: -i.daysUntil })),
    coverage: { completeHistory: false, snapshotAt: at, snapshotAgeDays: age, historyEvents: events.length, appEvents: valid.length, latestActivityAt, sources: ["local-kitchen-copy", "verified-kitchen-history", "local-shopping-activity", ...valid.some((e) => e.source === "app") ? ["app-activity"] : []], caveats: ["The kitchen is a dated local copy; local activity does not sync to the iOS app.", "Inventory observations and list selections are not purchases.", "No inference of allergies, health conditions or individual eating.", ...age !== null && age > 2 ? ["Kitchen quantities and shelf-life state may be stale; check before relying on them."] : [], ...!policy.enabled ? ["Learning is paused; earlier evidence and current kitchen state remain available."] : []] }
  };
}
function brainGroceries(brain, h, options = {}) {
  if (brain.householdId !== h.id) throw new Error("Brain scope mismatch");
  const limit = Math.max(1, Math.min(10, options.limit ?? 5)), excluded = new Set(options.excludeIds ?? []);
  const covered = (query) => h.list.some((i) => [i.name, i.savedProduct?.name, i.source?.originalName].some((n) => n && brainKey(query).includes(brainKey(n))));
  const pool = brain.groceries.items.filter((i) => !excluded.has(i.id) && !covered(i.query) && !h.list.some((l) => i.kitchenId && l.sourceKitchenId === i.kitchenId));
  const items = pool.slice(0, limit), past = brain.coverage.snapshotAgeDays;
  const intro = items.length ? `Here ${items.length === 1 ? "is one idea" : `are ${items.length} ideas`} from your kitchen history. I\u2019ve left out what\u2019s already on your list.` : "That\u2019s all I can support from your available history. Your existing list is covered.";
  return { ...brain.groceries, items, requestedCount: limit, hasMore: pool.length > limit, summary: intro + (past !== null && past > 2 ? " Your kitchen copy is older, so check what\u2019s left." : "") };
}

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/founder-shop/assistant.ts
function assistantItems(h, suggestions, now) {
  const kitchen = h.kitchen.map((k) => {
    const expiry = shelfLifeExpiry(k, now);
    const timing = expiry.date ? `${expiry.status === "date-recorded" ? "Recorded date" : "Estimated shelf life"} \xB7 ${expiry.date}` : null;
    return {
      kind: "kitchen",
      id: "kitchen:" + k.id,
      name: k.name,
      query: k.name,
      reason: timing ?? (k.low ? "Saved amount: " + k.remaining : k.remaining ? "In your kitchen \xB7 " + k.remaining : "In your kitchen"),
      confirmedLow: k.low === true,
      emoji: k.source?.imageEmoji ?? null,
      imageUrl: k.source?.imageUrl ?? null,
      kitchenId: k.id,
      remaining: k.remaining,
      low: k.low,
      location: k.location,
      deadline: expiry.date,
      daysUntil: expiry.daysUntil,
      dateBasis: expiry.status
    };
  });
  return [...kitchen, ...suggestions.map((s) => ({ ...s, kind: "restock", id: "suggestion:" + s.id, confirmedLow: false }))];
}

// ../../2026-09-23/trepo-agentic-shopping/outputs/trepo-shopping/lib/founder-shop/domain.ts
var short = (max) => external_exports.string().trim().min(1).max(max);
var addressSchema = external_exports.object({ name: short(100), line1: short(160), line2: external_exports.string().trim().max(100), city: short(100), region: external_exports.string().trim().regex(/^[A-Za-z]{2}$/).transform((s) => s.toUpperCase()), postalCode: external_exports.string().trim().regex(/^\d{5}(?:-\d{4})?$/), country: external_exports.literal("US") }).strict();
var checkoutClientTimingCommandSchema = external_exports.object({ type: external_exports.literal("recordCheckoutTiming"), checkoutId: external_exports.string().uuid(), elapsedMs: external_exports.number().int().min(0).max(864e5) }).strict();
var commandSchema = external_exports.discriminatedUnion("type", [
  checkoutClientTimingCommandSchema,
  external_exports.object({ type: external_exports.literal("prepareCheckout") }).strict(),
  external_exports.object({ type: external_exports.literal("finishCheckout"), checkoutId: external_exports.string().uuid() }).strict(),
  external_exports.object({ type: external_exports.literal("buyCheckout"), checkoutId: external_exports.string().uuid(), fingerprint: external_exports.string().regex(/^[a-f0-9]{64}$/) }).strict(),
  external_exports.object({ type: external_exports.literal("cancelCheckout"), checkoutId: external_exports.string().uuid() }).strict(),
  external_exports.object({ type: external_exports.literal("saveAddress"), address: addressSchema }).strict(),
  external_exports.object({ type: external_exports.literal("discoverItems"), items: external_exports.array(external_exports.object({ name: short(160), quantity: external_exports.number().int().min(1).max(20) }).strict()).max(20), message: short(1e3).optional(), expandSearch: external_exports.boolean().optional() }).strict(),
  external_exports.object({ type: external_exports.literal("retryDiscovery"), discoveryId: short(180), expandSearch: external_exports.boolean().optional() }).strict(),
  external_exports.object({ type: external_exports.literal("dismissDiscovery"), discoveryId: short(180) }).strict(),
  external_exports.object({ type: external_exports.literal("selectDiscovery"), discoveryId: short(180), offerId: short(180), strategy: external_exports.enum(["keep", "preview"]), selectionMode: external_exports.literal("add").optional() }).strict(),
  external_exports.object({ type: external_exports.literal("applyTransfer"), transferId: short(180) }).strict(),
  external_exports.object({ type: external_exports.literal("previewStore"), storeId: short(180) }).strict(),
  external_exports.object({ type: external_exports.literal("recommendStores") }).strict(),
  external_exports.object({ type: external_exports.literal("buildCart"), items: external_exports.array(external_exports.object({ name: short(160), quantity: external_exports.number().int().min(1).max(20) }).strict()).max(40), message: short(1e3).optional(), includeList: external_exports.boolean().optional() }).strict(),
  external_exports.object({ type: external_exports.literal("chooseAlternative"), proposalId: short(180), lineId: short(180), offerId: short(180), selectionMode: external_exports.literal("add").optional() }).strict(),
  external_exports.object({ type: external_exports.literal("selectCart"), proposalId: short(180), optionId: external_exports.enum(["one-store", "best-match"]) }).strict(),
  external_exports.object({ type: external_exports.literal("prepareCart") }).strict(),
  external_exports.object({ type: external_exports.literal("refreshMatches") }).strict(),
  external_exports.object({ type: external_exports.literal("chat"), message: short(1e3) }).strict(),
  external_exports.object({ type: external_exports.literal("addCandidate"), candidateId: short(180) }).strict(),
  external_exports.object({ type: external_exports.literal("setQuantity"), lineId: short(180), quantity: external_exports.number().int().min(1).max(20) }).strict(),
  external_exports.object({ type: external_exports.literal("removeLine"), lineId: short(180) }).strict(),
  external_exports.object({ type: external_exports.literal("resolveItem"), listItemId: short(180), candidateId: short(180), acknowledgeDifferences: external_exports.boolean() }).strict(),
  external_exports.object({ type: external_exports.literal("skipItem"), listItemId: short(180) }).strict(),
  external_exports.object({ type: external_exports.literal("order"), quoteId: external_exports.string().uuid(), fingerprint: external_exports.string().regex(/^[a-f0-9]{64}$/), idempotencyKey: external_exports.string().uuid() }).strict(),
  external_exports.object({ type: external_exports.literal("reconcile"), orderId: external_exports.string().uuid() }).strict(),
  external_exports.object({ type: external_exports.literal("createReimbursement"), orderId: external_exports.string().uuid() }).strict(),
  external_exports.object({ type: external_exports.literal("checkPayment"), orderId: external_exports.string().uuid() }).strict()
]);
var commandEnvelope = external_exports.object({ expectedRevision: external_exports.number().int().min(0), command: commandSchema }).strict();
var cents = external_exports.number().int().min(0).max(1e7);
var receiptSchema = external_exports.object({ currency: external_exports.literal("USD"), subtotalCents: cents, deliveryCents: cents, serviceCents: cents, taxCents: cents, tipCents: cents, bagCents: cents, discountCents: cents, totalCents: cents }).strict();
var safeLink = (value) => {
  try {
    const u = new URL(value);
    return u.protocol === "https:" && !u.username && !u.password;
  } catch {
    return false;
  }
};
var candidateSchema = external_exports.object({ id: short(180), name: short(220), brand: external_exports.string().max(100), size: external_exports.string().max(100), variant: external_exports.string().max(100).optional(), imageUrl: external_exports.string().max(1800).refine((v) => /^\/grocery\/[a-z0-9-]+\.svg$/.test(v) || safeLink(v)).nullable(), priceCents: cents.nullable(), sourceUrl: external_exports.string().max(1800).refine(safeLink).nullable(), reason: short(500), attributes: external_exports.array(external_exports.string().max(80)).max(30), allergens: external_exports.array(external_exports.string().max(80)).max(30).nullable(), availability: external_exports.enum(["simulated", "unknown"]) }).strict();
function shopContext(h, now = Date.now(), brain) {
  if (brain?.householdId !== h.id) brain = void 0;
  const scopedBrain = brain;
  if (!brain?.enabled) brain = void 0;
  const suggestions = [];
  for (let page = 0; page < 3; page++) {
    const options = { limit: 10, excludeIds: suggestions.map((i) => i.id) };
    const batch = scopedBrain ? brainGroceries(scopedBrain, h, options) : restockSuggestions(h, options, now);
    suggestions.push(...batch.items);
    if (!batch.hasMore) break;
  }
  const context = {
    sourceRevision: h.purchaseRevision,
    source: h.provenance?.kind ?? "synthetic-demo",
    retrievedAt: h.provenance?.retrievedAt ?? null,
    kitchen: h.kitchen.slice(0, 150).map((k) => ({ id: k.id, name: k.name, brand: k.preference.brand, remaining: k.remaining, location: k.location, low: k.low, emoji: k.source?.imageEmoji ?? null, imageUrl: k.source?.imageUrl ?? (k.image.startsWith("/grocery/") ? k.image : null) })),
    list: h.list.map((i) => ({ id: i.id, name: i.name, quantity: i.quantity, brand: i.preference.brand, variant: i.preference.variant, size: i.preference.size, imageUrl: i.source?.imageUrl ?? null })),
    preferences: [...(h.shoppingRequestPreferences ?? []).map((p) => ({ text: `${p.original}: ${p.preferred}`, basis: "user-stated" })), ...h.list.filter((i) => i.preference.brand || i.preference.variant || i.preference.size).map((i) => ({ text: `${i.name}: ${[i.preference.brand, i.preference.variant, i.preference.size].filter(Boolean).join(" \xB7 ")}`, basis: "saved-product" })), ...(brain?.profile.products ?? []).filter((p) => p.eligible).slice(0, 8).map((p) => ({ text: `Recorded product context: ${[p.name, p.variant, p.size].filter(Boolean).join(" \xB7 ")}${p.evidence.selectedDays ? `; chosen on ${p.evidence.selectedDays} days` : `; observed across ${p.evidence.distinctWeeks} weeks`}. Use as an option, not a restriction.`, basis: "observed-pattern" })), ...(brain?.profile.patterns ?? []).slice(0, 10).map((p) => ({ text: `${p.value} appears across ${p.productCount} recorded products${p.distinctWeeks ? ` in ${p.distinctWeeks} weeks` : ""}; an observation, not a confirmed preference.`, basis: "observed-pattern" }))].slice(0, 40),
    suggestions: suggestions.map((i) => ({ id: i.id, name: i.name, query: i.query, reason: i.reason, emoji: i.emoji ?? h.onlineShopping?.profile.observations.find((o) => o.subjectActorId === h.onlineShopping?.profile.subjectActorId && o.attribution === "recorded-actor" && o.name.toLowerCase() === i.name.toLowerCase() && o.imageEmoji)?.imageEmoji ?? null, imageUrl: h.kitchen.find((k) => k.id === i.kitchenId)?.source?.imageUrl ?? null })),
    caveat: "Kitchen entries and recorded choices are context, not proof of current stock or dietary needs. All prices, delivery times and receipts in this pilot are simulated.",
    familiarProducts: (brain?.profile.products ?? []).slice(0, 200).map((p) => ({ name: p.name, brand: p.brand ?? "", variant: p.variant ?? "", size: p.size ?? "" })),
    ...scopedBrain ? { brain: { generatedAt: scopedBrain.generatedAt, enabled: scopedBrain.enabled, historyEvents: scopedBrain.coverage.historyEvents, useSoon: scopedBrain.useSoon.slice(0, 5).map((i) => ({ name: i.name, date: i.date, basis: i.basis })) } } : {}
  };
  context.assistantItems = assistantItems(h, context.suggestions, now);
  context.recipes = h.savedRecipes ?? [];
  context.receiptStores = scopedBrain?.enabled === false ? [] : h.receiptStores ?? [];
  context.shoppingSignals = {
    preferences: context.preferences.map((p) => ({ ...p, text: p.text.slice(0, 500) })),
    products: (brain?.profile.products ?? []).slice(0, 200).map((p) => ({ name: p.name, brand: p.brand ?? "", variant: p.variant ?? "", size: p.size ?? "", purchaseDays: p.evidence.explicitPurchasedDays, consumedDays: p.evidence.explicitConsumedDays, selectedDays: p.evidence.selectedDays ?? 0, eligible: p.eligible, currentItems: p.evidence.currentItems, distinctDays: p.evidence.distinctDays, distinctWeeks: p.evidence.distinctWeeks, discardedDays: p.evidence.explicitDiscardedDays })),
    recipes: (brain?.recipePatterns ?? []).filter((p) => p.liked || p.cooked || p.disliked).slice(0, 12).map((p) => ({ style: p.tag.slice(0, 100), liked: p.liked, cooked: p.cooked, disliked: p.disliked })),
    completeHistory: false
  };
  return context;
}

// work/context-cloud/node/shopping-projection.mjs
async function shoppingProjection(raw, seed, now = Date.now()) {
  const actor = "trepo:" + seed.userId;
  if (raw.bundle.snapshot.ownerId !== seed.userId || raw.brain.fence?.household !== seed.householdId) throw Error("scope");
  const { household: h } = await prepareSessionDraft(actor, raw.bundle, now);
  h.id = seed.draft;
  h.shoppingRequestPreferences = seed.preferences;
  const members = new Set(raw.bundle.allowedRecordOwnerIds);
  const activities = seed.activities.filter((a) => a.actorId === actor || a.visibility === "household" && (a.actorId === null || typeof a.actorId === "string" && members.has(a.actorId.replace(/^trepo:/, ""))));
  const choices = (seed.choices ?? []).filter((c) => c.selectedBy === actor).map((c) => ({ id: "legacy-choice:" + c.id, kind: "product-selected", occurredAt: c.selectedAt, receivedAt: h.provenance.retrievedAt, name: c.productName, brand: c.brand, entityId: c.productId, tags: [], minutes: null, rating: null, source: "shopping", actorId: actor, visibility: "personal" }));
  const brain = await buildHouseholdBrain(h, [...activities, ...choices], seed.policy, now);
  return shopContext(h, now, brain);
}

// work/context-cloud/node/project.mjs
var ContextError = class extends Error {
};
var text2 = (value, max = 240) => typeof value === "string" ? value.replace(/[\x00-\x1f\x7f]/g, " ").slice(0, max) : "";
var list = (value, max) => Array.isArray(value) ? value.slice(0, max) : [];
function projectContext(input, owner, scope, now = Date.now()) {
  if (input?.source !== "trepo-readonly-copy" || !Number.isFinite(Date.parse(input.retrievedAt)) || Date.parse(input.retrievedAt) > now + 6e4) throw new ContextError(503, "CONTEXT_UNAVAILABLE");
  const enabled = input.brain?.enabled === true;
  const identity2 = (p) => JSON.stringify([p.name, p.brand ?? "", p.variant ?? "", p.size ?? ""]);
  const signals = new Map(list(input.shoppingSignals?.products, 200).map((p) => [identity2(p), p]));
  const count2 = (v) => Number.isInteger(v) && v >= 0 ? Math.min(v, 1e4) : 0;
  const products = (rows) => list(rows, 200).map((p) => ({ name: text2(p.name), brand: text2(p.brand, 100), variant: text2(p.variant, 100), size: text2(p.size, 80) }));
  return {
    version: 1,
    owner,
    scope,
    status: "ready",
    dataSource: input.contextSource ?? "app-snapshot",
    fetchedAt: new Date(now).toISOString(),
    sourceRetrievedAt: input.retrievedAt,
    kitchen: list(input.kitchen, 150).map((p) => ({ name: text2(p.name), brand: text2(p.brand, 100), remaining: text2(p.remaining, 80), location: text2(p.location, 60), low: typeof p.low === "boolean" ? p.low : null })),
    list: list(input.list, 100).map((p) => ({ ...products([p])[0], quantity: Number.isInteger(p.quantity) ? p.quantity : null })),
    preferences: list(input.preferences, 40).filter((p) => ["user-stated", "saved-product", ...enabled ? ["observed-pattern"] : []].includes(p.basis)).map((p) => ({ text: text2(p.text, 500), basis: p.basis })),
    familiarProducts: enabled ? products(input.familiarProducts).map((p) => {
      const s = signals.get(identity2(p));
      return { ...p, ...s ? { evidence: { purchaseDays: count2(s.purchaseDays), consumedDays: count2(s.consumedDays), selectedDays: count2(s.selectedDays), currentItems: count2(s.currentItems), distinctDays: count2(s.distinctDays), distinctWeeks: count2(s.distinctWeeks), discardedDays: count2(s.discardedDays) }, eligible: s.eligible === true } : {} };
    }) : [],
    receiptStores: enabled ? list(input.receiptStores, 40).map((s) => ({ name: text2(s.name, 180), receiptCount: count2(s.receiptCount), lastSeenAt: text2(s.lastSeenAt, 60), basis: "receipt-history" })).filter((s) => s.name && s.receiptCount) : [],
    suggestions: enabled ? list(input.suggestions, 20).map((p) => ({ name: text2(p.name), query: text2(p.query, 120), reason: text2(p.reason, 350) })) : [],
    recipes: list(input.recipes, 30).map((p) => ({ title: text2(p.title), ingredients: list(p.ingredients, 40).map((v) => text2(v, 180)) })),
    brain: { enabled, generatedAt: text2(input.brain?.generatedAt, 60), historyEvents: Number.isInteger(input.brain?.historyEvents) ? input.brain.historyEvents : 0 },
    caveat: "This dated read-only Trepo context uses the shopping app\u2019s household brain rules and controls. Recorded products are evidence, not confirmed favorites, current stock, purchases, allergies, or dietary requirements. Explicit requests take priority. DoorDash verifies current catalog prices separately."
  };
}

// work/context-cloud/node/handler.mjs
var privacyKey = (brain) => JSON.stringify({ generation: brain.fence?.generation, members_hash: brain.fence?.members_hash, states: [...brain.fence?.states ?? []].sort((a, b) => String(a[0]).localeCompare(String(b[0]))) });
var response = (statusCode, value) => ({ statusCode, headers: { "content-type": "application/json", "cache-control": "no-store", "x-content-type-options": "nosniff" }, body: JSON.stringify(value) });
function cacheEligible(cached, raw, owner, now) {
  return !!cached && cached.owner === owner && cached.context?.scope === raw.context.scope && Number.isFinite(cached.at) && cached.at <= now + 6e4 && now - cached.at < 15 * 60 * 1e3 && raw.brain.context?.coverage?.learning_paused === false && cached.anchor === privacyKey(raw.brain);
}
function mergeContext(current, historical) {
  const listed = new Set(current.list.map((x) => x.name.toLowerCase()));
  const preferences = [...current.preferences, ...historical.preferences];
  return {
    ...historical,
    ...current,
    preferences: [...new Map(preferences.map((p) => [p.text, p])).values()].slice(0, 40),
    familiarProducts: historical.familiarProducts,
    suggestions: historical.suggestions.filter((x) => !listed.has(x.name.toLowerCase())),
    brain: { ...historical.brain, source: "shopping-model+household-brain-v2", completeHistory: false }
  };
}
function createHandler({ config, invoke, readState, readCache = async () => null, writeCache = async () => {
}, project = shoppingProjection, clock = Date.now }) {
  return async (event, execution = {}) => {
    const started = clock();
    try {
      const c = config();
      if (!event.requestContext && event.operation === "refresh" && Object.keys(event).every((k) => k === "operation")) {
        const state = await readState();
        let refreshed = 0, failed = 0;
        for (const [owner2, seed] of Object.entries(state.entries)) {
          let failure = null;
          for (let attempt = 1; attempt <= 2; attempt++) {
            try {
              if (execution.getRemainingTimeInMillis && execution.getRemainingTimeInMillis() < 32000) {
                throw Object.assign(Error('deadline'), { code: 'DEADLINE_RESERVE' });
              }
              const raw2 = await invoke({ operation: "snapshot", owner: owner2, token: c.token });
              if (raw2.statusCode !== 200) throw Object.assign(Error('source'), { code: raw2.code, status: raw2.statusCode });
              if (raw2.owner !== owner2 || raw2.bundle?.snapshot.ownerId !== seed.userId || raw2.brain?.fence?.household !== seed.householdId) {
                throw Object.assign(Error('scope'), { code: 'SCOPE_CHANGED' });
              }
              const mayProject = raw2.brain.context?.coverage?.learning_paused === false && seed.anchor === privacyKey(raw2.brain);
              const shopping = mayProject ? await project(raw2, seed, clock()) : null;
              const value2 = shopping ? projectContext(shopping, owner2, raw2.context.scope, clock()) : raw2.context;
              const checked2 = await invoke({ operation: "validate", owner: owner2, token: c.token, fence: raw2.brain.fence ?? null, dietary: raw2.dietary });
              if (checked2.statusCode !== 200 || checked2.current !== true) {
                throw Object.assign(Error('changed'), { code: checked2.code ?? 'CONTEXT_CHANGED', status: checked2.statusCode });
              }
              await writeCache(owner2, { owner: owner2, at: clock(), anchor: privacyKey(raw2.brain), context: value2 });
              refreshed++;
              failure = null;
              break;
            } catch (error) {
              const allowed = new Set(['BRAIN_UNAVAILABLE', 'CONTEXT_UNAVAILABLE', 'CONTEXT_CHANGED',
                'UNAUTHORIZED', 'FORBIDDEN', 'SCOPE_CHANGED', 'DEADLINE_RESERVE']);
              failure = allowed.has(error.code) ? error.code : 'REFRESH_UNAVAILABLE';
              const transient = ['BRAIN_UNAVAILABLE', 'CONTEXT_UNAVAILABLE', 'CONTEXT_CHANGED'].includes(failure)
                && ![401, 403, 404, 428].includes(error.status);
              // A fresh snapshot is required: never reuse a failed validation fence.
              if (!transient || attempt === 2) break;
            }
          }
          if (failure) {
            failed++;
            console.warn(JSON.stringify({event: 'trepo_context_owner_refresh_failed', owner: owner2, code: failure}));
          }
        }
        // A per-owner failure is observable without making Lambda replay every
        // successful owner. Scheduled refreshes retry failed owners next run.
        console.info(JSON.stringify({
          _aws: {Timestamp: clock(), CloudWatchMetrics: [{Namespace: 'Trepo/UX', Dimensions: [[]],
            Metrics: [{Name: 'ContextProjectionOwnersFailed', Unit: 'Count'}]}]},
          ContextProjectionOwnersFailed: failed, event: "trepo_context_refresh", refreshed, failed,
          milliseconds: clock() - started
        }));
        return { refreshed, failed };
      }
      const given = Buffer.from(event.headers?.authorization ?? ""), expected = Buffer.from("Bearer " + c.token);
      if (given.length !== expected.length || !timingSafeEqual(given, expected)) return response(401, { status: "unavailable", code: "UNAUTHORIZED" });
      const match = /^\/v1\/context\/([a-f0-9]{64})$/.exec(event.rawPath ?? "");
      if (event.requestContext?.http?.method !== "GET" || !match || event.rawQueryString || event.body) return response(404, { status: "unavailable", code: "NOT_FOUND" });
      const owner = match[1];
      const raw = await invoke({ operation: "current", owner, token: c.token });
      if (raw.statusCode !== 200) return response(raw.statusCode ?? 503, { status: "unavailable", code: raw.code ?? "CONTEXT_UNAVAILABLE" });
      if (raw.owner !== owner || raw.context?.owner !== owner || raw.context?.status !== "ready") throw Error("scope");
      const cached = await readCache(owner).catch(() => null);
      const eligible = cacheEligible(cached, raw, owner, clock());
      const value = eligible ? mergeContext(raw.context, cached.context) : raw.context;
      const checked = await invoke({ operation: "validate", owner, token: c.token, fence: raw.brain.fence ?? null, dietary: raw.dietary });
      if (checked.statusCode !== 200 || checked.current !== true) return response(checked.statusCode ?? 503, { status: "unavailable", code: checked.code ?? "CONTEXT_CHANGED" });
      if (Buffer.byteLength(JSON.stringify(value)) > 28e4) throw Error("size");
      console.info(JSON.stringify({ event: "trepo_context_cloud", status: "ready", milliseconds: clock() - started, shopping_model: eligible }));
      return response(200, value);
    } catch (e) {
      console.warn(JSON.stringify({ event: "trepo_context_cloud", status: "unavailable", kind: e.constructor?.name ?? "Error", milliseconds: clock() - started }));
      if (!event.requestContext && event.operation === "refresh") throw Error("Context refresh failed");
      return response(503, { status: "unavailable", code: "CONTEXT_UNAVAILABLE" });
    }
  };
}

// work/context-cloud/node/runtime.mjs
var lambda = new LambdaClient({ maxAttempts: 1 });
var s3 = new S3Client({ maxAttempts: 1 });
var handler = createHandler({
  config: () => JSON.parse(process.env.CONTEXT_CONFIG),
  async invoke(request) {
    const result = await lambda.send(new InvokeCommand({ FunctionName: process.env.SOURCE_FUNCTION, Payload: Buffer.from(JSON.stringify(request)) }), { abortSignal: AbortSignal.timeout(14e3) });
    if (result.FunctionError) throw Error("source");
    return JSON.parse(Buffer.from(result.Payload).toString());
  },
  async readCache(owner) {
    const response2 = await s3.send(new GetObjectCommand({ Bucket: process.env.STATE_BUCKET, Key: "cache/" + owner + ".json" }), { abortSignal: AbortSignal.timeout(2e3) });
    if (response2.ContentLength > 3e5) throw Error("cache");
    return JSON.parse(await response2.Body.transformToString());
  },
  async writeCache(owner, value) {
    await s3.send(new PutObjectCommand({ Bucket: process.env.STATE_BUCKET, Key: "cache/" + owner + ".json", Body: JSON.stringify(value), ServerSideEncryption: "AES256", ContentType: "application/json" }), { abortSignal: AbortSignal.timeout(3e3) });
  },
  async readState() {
    const response2 = await s3.send(new GetObjectCommand({ Bucket: process.env.STATE_BUCKET, Key: process.env.STATE_KEY }), { abortSignal: AbortSignal.timeout(3e3) });
    if (response2.ContentLength > 2e6) throw Error("state");
    return JSON.parse(await response2.Body.transformToString());
  }
});
export {
  handler, createHandler, privacyKey
};
