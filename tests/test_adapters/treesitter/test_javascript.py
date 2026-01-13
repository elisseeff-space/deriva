"""Tests for JavaScript language extractor."""

from __future__ import annotations

from deriva.adapters.treesitter import TreeSitterManager


class TestJavaScriptTypes:
    """Tests for JavaScript type extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_extracts_class(self):
        """Should extract ES6 class definition."""
        source = """
class UserService {
    constructor() {
        this.users = [];
    }
}
"""
        types = self.manager.extract_types(source, language="javascript")

        assert len(types) >= 1
        service = next((t for t in types if t.name == "UserService"), None)
        assert service is not None
        assert service.kind == "class"

    def test_extracts_class_with_extends(self):
        """Should extract class inheritance."""
        source = """
class Admin extends User {
    constructor() {
        super();
    }
}
"""
        types = self.manager.extract_types(source, language="javascript")

        admin = next((t for t in types if t.name == "Admin"), None)
        assert admin is not None
        assert admin.kind == "class"

    def test_extracts_function_declaration(self):
        """Should extract function declaration."""
        source = """
function processData(data) {
    return data.map(x => x * 2);
}
"""
        types = self.manager.extract_types(source, language="javascript")

        func = next((t for t in types if t.name == "processData"), None)
        assert func is not None
        assert func.kind == "function"

    def test_extracts_async_function(self):
        """Should detect async function declarations."""
        source = """
async function fetchUser(id) {
    const response = await fetch(`/api/users/${id}`);
    return response.json();
}
"""
        types = self.manager.extract_types(source, language="javascript")

        func = next((t for t in types if t.name == "fetchUser"), None)
        assert func is not None
        assert func.is_async is True

    def test_extracts_exported_function(self):
        """Should extract exported functions."""
        source = """
export function helper() {
    return 42;
}

export async function asyncHelper() {
    return await Promise.resolve(42);
}
"""
        types = self.manager.extract_types(source, language="javascript")

        names = {t.name for t in types}
        assert "helper" in names
        assert "asyncHelper" in names

    def test_extracts_exported_class(self):
        """Should extract exported class."""
        source = """
export class Service {
    run() {
        return 'running';
    }
}
"""
        types = self.manager.extract_types(source, language="javascript")

        service = next((t for t in types if t.name == "Service"), None)
        assert service is not None

    def test_extracts_arrow_function_variable(self):
        """Should extract const arrow functions as types."""
        source = """
const add = (a, b) => a + b;

const multiply = (x, y) => {
    return x * y;
};
"""
        types = self.manager.extract_types(source, language="javascript")

        # Arrow functions assigned to const should be extracted
        multiply = next((t for t in types if t.name == "multiply"), None)
        assert multiply is not None
        assert multiply.kind == "function"

    def test_extracts_async_arrow_function(self):
        """Should extract async arrow functions."""
        source = """
const fetchData = async (url) => {
    const res = await fetch(url);
    return res.json();
};
"""
        types = self.manager.extract_types(source, language="javascript")
        fetch_func = next((t for t in types if t.name == "fetchData"), None)

        assert fetch_func is not None
        assert fetch_func.is_async is True


class TestJavaScriptMethods:
    """Tests for JavaScript method extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_extracts_class_methods(self):
        """Should extract methods from class."""
        source = """
class Calculator {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}
"""
        methods = self.manager.extract_methods(source, language="javascript")

        names = {m.name for m in methods}
        assert "add" in names
        assert "subtract" in names

    def test_extracts_constructor(self):
        """Should extract constructor method."""
        source = """
class Service {
    constructor(config) {
        this.config = config;
    }
}
"""
        methods = self.manager.extract_methods(source, language="javascript")

        constructor = next((m for m in methods if m.name == "constructor"), None)
        assert constructor is not None

    def test_extracts_async_method(self):
        """Should detect async class methods."""
        source = """
class ApiClient {
    async get(url) {
        return await fetch(url);
    }

    async post(url, data) {
        return await fetch(url, { method: 'POST', body: data });
    }
}
"""
        methods = self.manager.extract_methods(source, language="javascript")

        async_methods = [m for m in methods if m.is_async]
        assert len(async_methods) >= 1

    def test_extracts_static_method(self):
        """Should extract static methods."""
        source = """
class MathUtils {
    static square(x) {
        return x * x;
    }

    static cube(x) {
        return x * x * x;
    }
}
"""
        methods = self.manager.extract_methods(source, language="javascript")

        names = {m.name for m in methods}
        assert "square" in names or "cube" in names

    def test_extracts_getter_setter(self):
        """Should extract getters and setters."""
        source = """
class Person {
    get name() {
        return this._name;
    }

    set name(value) {
        this._name = value;
    }
}
"""
        methods = self.manager.extract_methods(source, language="javascript")

        names = {m.name for m in methods}
        assert "name" in names

    def test_extracts_top_level_function(self):
        """Should extract top-level functions."""
        source = """
function helper(x) {
    return x * 2;
}
"""
        methods = self.manager.extract_methods(source, language="javascript")

        helper = next((m for m in methods if m.name == "helper"), None)
        assert helper is not None
        assert helper.class_name is None

    def test_extracts_arrow_function_method(self):
        """Should extract arrow function class fields."""
        source = """
class Handler {
    handleClick = () => {
        console.log('clicked');
    };

    handleSubmit = async (data) => {
        await this.submit(data);
    };
}
"""
        methods = self.manager.extract_methods(source, language="javascript")
        names = {m.name for m in methods}

        # At minimum, one of the handlers should be extracted
        assert "handleClick" in names or "handleSubmit" in names


class TestJavaScriptImports:
    """Tests for JavaScript import extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_extracts_default_import(self):
        """Should extract default import."""
        source = """
import React from 'react';
"""
        imports = self.manager.extract_imports(source, language="javascript")

        assert len(imports) >= 1
        react_import = next((i for i in imports if "react" in i.module), None)
        assert react_import is not None

    def test_extracts_named_imports(self):
        """Should extract named imports."""
        source = """
import { useState, useEffect, useCallback } from 'react';
"""
        imports = self.manager.extract_imports(source, language="javascript")

        assert len(imports) >= 1

    def test_extracts_namespace_import(self):
        """Should extract namespace imports."""
        source = """
import * as utils from './utils';
"""
        imports = self.manager.extract_imports(source, language="javascript")

        assert len(imports) >= 1

    def test_extracts_side_effect_import(self):
        """Should extract side-effect imports."""
        source = """
import './styles.css';
import 'polyfill';
"""
        imports = self.manager.extract_imports(source, language="javascript")

        assert len(imports) >= 1

    def test_extracts_mixed_imports(self):
        """Should handle mixed import styles."""
        source = """
import React, { Component, useState } from 'react';
"""
        imports = self.manager.extract_imports(source, language="javascript")

        assert len(imports) >= 1

    def test_extracts_require_statement(self):
        """Should extract CommonJS require statements."""
        source = """
const express = require('express');
const { Router } = require('express');
"""
        imports = self.manager.extract_imports(source, language="javascript")

        # CommonJS require should be captured as imports
        modules = [i.module for i in imports]
        assert any("express" in m for m in modules)


class TestJavaScriptEdgeCases:
    """Tests for edge cases in JavaScript extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_handles_empty_file(self):
        """Should handle empty file."""
        types = self.manager.extract_types("", language="javascript")
        assert types == []

    def test_handles_jsx_syntax(self):
        """Should handle JSX syntax."""
        source = """
function Component() {
    return <div>Hello</div>;
}
"""
        types = self.manager.extract_types(source, language="javascript")
        component = next((t for t in types if t.name == "Component"), None)

        assert component is not None
        assert component.kind == "function"

    def test_handles_template_literals(self):
        """Should handle template literals in code."""
        source = """
function greet(name) {
    return `Hello, ${name}!`;
}
"""
        types = self.manager.extract_types(source, language="javascript")

        assert len(types) >= 1

    def test_handles_private_fields(self):
        """Should handle private class fields."""
        source = """
class Counter {
    #count = 0;

    increment() {
        this.#count++;
    }
}
"""
        methods = self.manager.extract_methods(source, language="javascript")
        increment = next((m for m in methods if m.name == "increment"), None)

        assert increment is not None
        assert increment.class_name == "Counter"
